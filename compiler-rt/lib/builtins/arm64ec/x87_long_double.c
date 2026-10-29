//===-- x87_long_double.c - ARM64EC soft x87 `long double` helpers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ARM64EC is ABI compatible with x86-64 Windows, where `long double` is the
// x87 80-bit extended format held in a 16-byte, 16-byte-aligned slot.  There
// is no x87 FPU on ARM64, so the compiler keeps such values in memory and
// copies them around with integer instructions only; whenever a computation
// has to be carried out, it converts the value to `double`, computes in
// `double`, and converts the result back.
//
// The compiler softens every operation on the type to a call to one of the
// `__arm64ec_<op>xf<n>` helpers below, which take and return the 16-byte image
// by value.  The prefix marks them as the ARM64EC ones, which are not the exact
// x87 soft-float helpers of libgcc: all arithmetic is carried out in `double`,
// while comparisons are exact as they only inspect the image.  The older
// `__arm64ec_load_x87_long_double()` and `__arm64ec_store_x87_long_double()`
// are the equivalent memory forms, which are kept for code that already makes
// use of them.
//
// The 16-byte slot (little endian, as used by both x86-64 and ARM64) is
//
//   offset  0: 64-bit significand, including the explicit integer bit
//   offset  8: 16-bit sign/exponent word: bit 15 is the sign, bits 14..0 is
//              the biased exponent
//   offset 10: 6 bytes of padding that are not read or written by the x87
//
// Note that the x87 extended format does not use a hidden bit: The
// significand of a normal number has bit 63 set, and for a denormal or zero
// bit 63 is clear.
//
//===----------------------------------------------------------------------===//

#include "../int_lib.h"

/// The register image of an x87 extended-precision value: the 64-bit
/// significand in the low half and the sign/exponent word in the high half.
/// `x86_fp80` is softened to this type by the backend.
typedef tu_int arm64ec_x87_reg_image;

/// The memory image of an x87 extended-precision value.  Deliberately declared
/// as integers so that neither the compiler nor this file can end up using
/// floating-point instructions on it.
typedef struct {
  uint64_t significand;
  uint16_t sign_exponent;
  uint16_t padding[3];
} arm64ec_x87_image;

typedef union {
  uint64_t u;
  double d;
} arm64ec_double_bits;

#define ARM64EC_X87_EXP_BIAS 16383
#define ARM64EC_X87_EXP_MAX 0x7fffu
#define ARM64EC_DBL_EXP_MAX 0x7ffu

/// The x87 real indefinite, which is what it produces for an invalid operand.
/// Its sign bit is always set.
static double arm64ec_indefinite_qnan(void) {
  arm64ec_double_bits v;
  v.u = 0xfff8000000000000ull;
  return v.d;
}

static double arm64ec_make_double(unsigned sign, uint32_t biased_exponent,
                                  uint64_t fraction) {
  arm64ec_double_bits v;
  v.u = ((uint64_t)sign << 63) | ((uint64_t)biased_exponent << 52) | fraction;
  return v.d;
}

/// Convert the parts of an x87 extended-precision image to `double`, rounding
/// to nearest with ties to even.  This matches what the x87 itself does for
/// 'fld m80; fstp m64' with the precision control set to 64 bits.
///
/// This works on the 64-bit significand and the sign/exponent word directly
/// rather than on a memory image, so that callers need not store anything and
/// the helpers that pass images by value can stay free of memory effects.
static double arm64ec_image_to_double(uint64_t m, uint16_t sign_exponent) {
  unsigned sign = (unsigned)(sign_exponent >> 15);
  unsigned e = (unsigned)(sign_exponent & 0x7fff);
  int exponent;
  uint64_t kept, dropped;

  if (e == 0x7fff) {
    if (m == 0x8000000000000000ull)
      // Infinity.
      return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX, 0);
    // A NaN whose integer bit is set is a real NaN: keep the top 52 bits of
    // its payload and make sure the quiet bit is set.  Anything else (a
    // pseudo-NaN or pseudo-infinity with the integer bit clear) is an invalid
    // encoding, for which the hardware produces the indefinite QNaN.
    if ((m >> 63) == 0)
      return arm64ec_indefinite_qnan();
    return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX,
                               ((m >> 11) & 0xfffffffffffffull) |
                                   0x8000000000000ull);
  }

  // Non-zero biased exponent with the integer bit clear is invalid.
  if (e != 0 && (m >> 63) == 0)
    return arm64ec_indefinite_qnan();

  // A pseudo-denormal (biased exponent zero with the integer bit set) is a
  // perfectly valid representation of m * 2^-16445 and behaves like the
  // normal number with biased exponent one.  A denormalized number is far
  // below the range of `double` and therefore underflows to zero.
  if (e == 0) {
    e = (unsigned)(m >> 63);
    if (e == 0) // zero or denormal, both of which become a signed zero
      return arm64ec_make_double(sign, 0, 0);
  }

  // value == m * 2^(exponent - 63), with bit 63 of m set
  exponent = (int)e - ARM64EC_X87_EXP_BIAS;

  if (exponent > 1023)
    return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX, 0);

  if (exponent >= -1022) {
    // A normal `double`: round the 64-bit significand to 53 bits.
    kept = m >> 11;
    dropped = m & 0x7ffu;
    if (dropped > 0x400u || (dropped == 0x400u && (kept & 1)))
      if (++kept == 0x20000000000000ull) {
        kept >>= 1;
        if (++exponent > 1023)
          return arm64ec_make_double(sign, ARM64EC_DBL_EXP_MAX, 0);
      }
    return arm64ec_make_double(sign, (uint32_t)(exponent + 1023),
                               kept & 0xfffffffffffffull);
  }

  // A subnormal `double`: the result is
  // round(m * 2^(exponent + 1011)) units of 2^-1074.
  int shift = -(exponent + 1011);
  uint64_t k;

  if (shift > 64)
    k = 0;
  else if (shift == 64) {
    // m is in [2^63, 2^64), so the quotient is in [0.5, 1) and only a value
    // strictly greater than one half rounds up.
    k = m > 0x8000000000000000ull ? 1 : 0;
  } else {
    uint64_t quotient = m >> shift;
    uint64_t remainder = m & ((1ull << shift) - 1);
    uint64_t half = 1ull << (shift - 1);
    if (remainder > half || (remainder == half && (quotient & 1)))
      ++quotient;
    k = quotient;
  }
  // 'k' may reach 2^52, which is exactly the smallest normal `double`; it
  // then carries into the exponent field on its own.
  return arm64ec_make_double(sign, 0, k);
}

/// Convert a `double` to the parts of an x87 extended-precision image.  Every
/// `double` is exactly representable in the x87 format, so this never rounds.
static void arm64ec_double_to_image(double value, uint64_t *significand,
                                    uint16_t *sign_exponent) {
  arm64ec_double_bits v;
  unsigned sign;
  unsigned e;
  uint64_t m;

  v.d = value;
  sign = (unsigned)(v.u >> 63);
  e = (unsigned)((v.u >> 52) & 0x7ff);
  m = v.u & 0xfffffffffffffull;

  if (e == ARM64EC_DBL_EXP_MAX) {
    if (m == 0)
      *significand = 0x8000000000000000ull; // infinity
    else
      // Map the 52-bit `double` payload onto the top bits of the 63-bit x87
      // fraction.  Like 'fld m64', quiet a signalling NaN.
      *significand = (m << 11) | 0xc000000000000000ull;
    *sign_exponent = (uint16_t)((sign << 15) | ARM64EC_X87_EXP_MAX);
  } else if (e == 0) {
    if (m == 0) {
      *significand = 0; // zero
      *sign_exponent = (uint16_t)(sign << 15);
    } else {
      // A subnormal `double` has to be normalized: it is
      // 1.f * 2^(p - 1074) where p is the index of its top set bit.
      int p = 63 - __builtin_clzll(m);
      *significand = m << (63 - p);
      *sign_exponent = (uint16_t)((sign << 15) | (unsigned)(p + 15309));
    }
  } else {
    // Shift the 52-bit `double` fraction up to the top of the 63-bit x87
    // fraction and add the explicit integer bit.
    *significand = (m << 11) | 0x8000000000000000ull;
    *sign_exponent = (uint16_t)((sign << 15) | (e + 15360));
  }
}

/// Convert an x87 extended-precision memory image to `double`.
double __arm64ec_load_x87_long_double(const arm64ec_x87_image *image) {
  return arm64ec_image_to_double(image->significand, image->sign_exponent);
}

/// Convert a `double` to an x87 extended-precision memory image.  Padding bits
/// are undefined.
void __arm64ec_store_x87_long_double(arm64ec_x87_image *image, double value) {
  arm64ec_double_to_image(value, &image->significand, &image->sign_exponent);
}

/// Convert a 16-byte x87 extended-precision image to `double`.  This is the
/// form the compiler calls; the image is passed by value in registers and its
/// six padding bytes are ignored.
double __arm64ec_x87ld_to_double(unsigned __int128 image) {
  return arm64ec_image_to_double((uint64_t)image, (uint16_t)(image >> 64));
}

/// Convert a `double` to a 16-byte x87 extended-precision image, with the
/// padding bits cleared.
unsigned __int128 __arm64ec_double_to_x87ld(double value) {
  uint64_t significand;
  uint16_t sign_exponent;

  arm64ec_double_to_image(value, &significand, &sign_exponent);
  return ((unsigned __int128)sign_exponent << 64) | significand;
}

//===----------------------------------------------------------------------===//
// Register-image helpers
//===----------------------------------------------------------------------===//

/// Convert a register image to `double`.
static double arm64ec_reg_to_double(arm64ec_x87_reg_image image) {
  return arm64ec_image_to_double((uint64_t)image, (uint16_t)(image >> 64));
}

/// Convert a `double` to a register image.  The padding bits are cleared.
static arm64ec_x87_reg_image arm64ec_double_to_reg(double value) {
  uint64_t significand;
  uint16_t sign_exponent;

  arm64ec_double_to_image(value, &significand, &sign_exponent);
  return ((arm64ec_x87_reg_image)sign_exponent << 64) | significand;
}

/// Test whether a register image is a NaN.  An encoding whose biased exponent
/// is neither zero nor all-ones and whose integer bit is clear is an invalid
/// encoding which the hardware treats as a NaN.
static int arm64ec_reg_is_nan(arm64ec_x87_reg_image image) {
  uint64_t significand = (uint64_t)image;
  unsigned biased_exponent = (unsigned)(image >> 64) & 0x7fffu;

  if (biased_exponent == ARM64EC_X87_EXP_MAX)
    return significand != 0x8000000000000000ull;
  return biased_exponent != 0 && (significand >> 63) == 0;
}

/// Compare two register images exactly, returning -1, 0 or 1.  The caller
/// guarantees that neither operand is a NaN.
static int arm64ec_reg_compare(arm64ec_x87_reg_image a,
                               arm64ec_x87_reg_image b) {
  uint64_t am = (uint64_t)a, bm = (uint64_t)b;
  unsigned ase = (unsigned)(a >> 64) & 0xffffu;
  unsigned bse = (unsigned)(b >> 64) & 0xffffu;
  unsigned asign = ase >> 15, bsign = bse >> 15;
  // A biased exponent of zero (a denormal or a pseudo-denormal) behaves as
  // exponent one, and a pseudo-denormal compares equal to the corresponding
  // normal encoding.
  unsigned ae = ase & 0x7fffu, be = bse & 0x7fffu;
  int magnitude;

  if (ae == 0)
    ae = 1;
  if (be == 0)
    be = 1;

  if (asign != bsign) {
    // The two zeroes compare equal.
    if (am == 0 && bm == 0)
      return 0;
    return asign ? -1 : 1;
  }

  if (ae != be)
    magnitude = ae < be ? -1 : 1;
  else if (am != bm)
    magnitude = am < bm ? -1 : 1;
  else
    magnitude = 0;

  return asign ? -magnitude : magnitude;
}

/// The `__le` form of a three-way comparison, which reports unordered operands
/// as greater than everything else, matching `__lesf2`/`__letf2`.
static int arm64ec_cmp_le(arm64ec_x87_reg_image a, arm64ec_x87_reg_image b) {
  if (arm64ec_reg_is_nan(a) || arm64ec_reg_is_nan(b))
    return 1;
  return arm64ec_reg_compare(a, b);
}

/// The `__ge` form of a three-way comparison, which reports unordered operands
/// as less than everything else, matching `__gesf2`/`__getf2`.
static int arm64ec_cmp_ge(arm64ec_x87_reg_image a, arm64ec_x87_reg_image b) {
  if (arm64ec_reg_is_nan(a) || arm64ec_reg_is_nan(b))
    return -1;
  return arm64ec_reg_compare(a, b);
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_addxf3(arm64ec_x87_reg_image a,
                                                       arm64ec_x87_reg_image b) {
  return arm64ec_double_to_reg(arm64ec_reg_to_double(a) +
                               arm64ec_reg_to_double(b));
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_subxf3(arm64ec_x87_reg_image a,
                                                       arm64ec_x87_reg_image b) {
  return arm64ec_double_to_reg(arm64ec_reg_to_double(a) -
                               arm64ec_reg_to_double(b));
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_mulxf3(arm64ec_x87_reg_image a,
                                                       arm64ec_x87_reg_image b) {
  return arm64ec_double_to_reg(arm64ec_reg_to_double(a) *
                               arm64ec_reg_to_double(b));
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_divxf3(arm64ec_x87_reg_image a,
                                                       arm64ec_x87_reg_image b) {
  return arm64ec_double_to_reg(arm64ec_reg_to_double(a) /
                               arm64ec_reg_to_double(b));
}

/// The `maxnum`/`minnum` forms, which return the ordering operand rather than
/// a NaN when exactly one of the two is a NaN, as `fmax`/`fmin` do.
COMPILER_RT_ABI arm64ec_x87_reg_image
__arm64ec_fmaxxf3(arm64ec_x87_reg_image a, arm64ec_x87_reg_image b) {
  return arm64ec_double_to_reg(__builtin_fmax(arm64ec_reg_to_double(a),
                                              arm64ec_reg_to_double(b)));
}

COMPILER_RT_ABI arm64ec_x87_reg_image
__arm64ec_fminxf3(arm64ec_x87_reg_image a, arm64ec_x87_reg_image b) {
  return arm64ec_double_to_reg(__builtin_fmin(arm64ec_reg_to_double(a),
                                              arm64ec_reg_to_double(b)));
}

COMPILER_RT_ABI arm64ec_x87_reg_image
__arm64ec_fmaxf3(arm64ec_x87_reg_image a, arm64ec_x87_reg_image b,
                 arm64ec_x87_reg_image c) {
  return arm64ec_double_to_reg(__builtin_fma(arm64ec_reg_to_double(a),
                                             arm64ec_reg_to_double(b),
                                             arm64ec_reg_to_double(c)));
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_powixf2(arm64ec_x87_reg_image a,
                                                        int b) {
  double result = 1.0;
  double base = arm64ec_reg_to_double(a);
  unsigned exponent = b < 0 ? -(unsigned)b : (unsigned)b;

  while (exponent != 0) {
    if (exponent & 1)
      result *= base;
    base *= base;
    exponent >>= 1;
  }

  return arm64ec_double_to_reg(b < 0 ? 1.0 / result : result);
}

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

COMPILER_RT_ABI int __arm64ec_eqxf2(arm64ec_x87_reg_image a,
                                    arm64ec_x87_reg_image b) {
  return arm64ec_cmp_le(a, b);
}

COMPILER_RT_ABI int __arm64ec_nexf2(arm64ec_x87_reg_image a,
                                    arm64ec_x87_reg_image b) {
  return arm64ec_cmp_le(a, b);
}

COMPILER_RT_ABI int __arm64ec_ltxf2(arm64ec_x87_reg_image a,
                                    arm64ec_x87_reg_image b) {
  return arm64ec_cmp_le(a, b);
}

COMPILER_RT_ABI int __arm64ec_lexf2(arm64ec_x87_reg_image a,
                                    arm64ec_x87_reg_image b) {
  return arm64ec_cmp_le(a, b);
}

COMPILER_RT_ABI int __arm64ec_gexf2(arm64ec_x87_reg_image a,
                                    arm64ec_x87_reg_image b) {
  return arm64ec_cmp_ge(a, b);
}

COMPILER_RT_ABI int __arm64ec_gtxf2(arm64ec_x87_reg_image a,
                                    arm64ec_x87_reg_image b) {
  return arm64ec_cmp_ge(a, b);
}

COMPILER_RT_ABI int __arm64ec_unordxf2(arm64ec_x87_reg_image a,
                                       arm64ec_x87_reg_image b) {
  return arm64ec_reg_is_nan(a) || arm64ec_reg_is_nan(b);
}

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_extendsfxf2(float a) {
  return arm64ec_double_to_reg((double)a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_extenddfxf2(double a) {
  return arm64ec_double_to_reg(a);
}

#if defined(__FLT16_MANT_DIG__)
COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_extendhfxf2(_Float16 a) {
  return arm64ec_double_to_reg((double)a);
}
#endif

COMPILER_RT_ABI float __arm64ec_truncxfsf2(arm64ec_x87_reg_image a) {
  return (float)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI double __arm64ec_truncxfdf2(arm64ec_x87_reg_image a) {
  return arm64ec_reg_to_double(a);
}

#if defined(__FLT16_MANT_DIG__)
COMPILER_RT_ABI _Float16 __arm64ec_truncxfhf2(arm64ec_x87_reg_image a) {
  return (_Float16)(float)arm64ec_reg_to_double(a);
}
#endif

// `__bf16` has no predefined macro of its own in C mode, but the type is always
// available on AArch64, so this helper is unconditional.
COMPILER_RT_ABI __bf16 __arm64ec_truncxfbf2(arm64ec_x87_reg_image a) {
  return (__bf16)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI int __arm64ec_fixxfsi(arm64ec_x87_reg_image a) {
  return (int)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI di_int __arm64ec_fixxfdi(arm64ec_x87_reg_image a) {
  return (di_int)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI ti_int __arm64ec_fixxfti(arm64ec_x87_reg_image a) {
  return (ti_int)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI unsigned __arm64ec_fixunsxfsi(arm64ec_x87_reg_image a) {
  return (unsigned)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI du_int __arm64ec_fixunsxfdi(arm64ec_x87_reg_image a) {
  return (du_int)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI tu_int __arm64ec_fixunsxfti(arm64ec_x87_reg_image a) {
  return (tu_int)arm64ec_reg_to_double(a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_floatsixf(si_int a) {
  return arm64ec_double_to_reg((double)a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_floatdixf(di_int a) {
  return arm64ec_double_to_reg((double)a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_floattixf(ti_int a) {
  return arm64ec_double_to_reg((double)a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_floatunsixf(su_int a) {
  return arm64ec_double_to_reg((double)a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_floatundixf(du_int a) {
  return arm64ec_double_to_reg((double)a);
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_floatuntixf(tu_int a) {
  return arm64ec_double_to_reg((double)a);
}

//===----------------------------------------------------------------------===//
// Conversions between x87 extended precision and IEEE binary128
//===----------------------------------------------------------------------===//

/// Convert a binary128 bit pattern to a floating-point value without invoking
/// any long double arithmetic.
typedef __float128 arm64ec_f128;

typedef union {
  arm64ec_f128 f;
  tu_int u;
} arm64ec_f128_bits;

COMPILER_RT_ABI arm64ec_f128
__arm64ec_extendxftf2(arm64ec_x87_reg_image image) {
  arm64ec_f128_bits result;
  uint64_t significand = (uint64_t)image;
  unsigned se = (unsigned)(image >> 64) & 0xffffu;
  unsigned sign = se >> 15;
  unsigned e = se & 0x7fffu;
  tu_int bits = (tu_int)sign << 127;

  if (e == ARM64EC_X87_EXP_MAX) {
    if (significand == 0x8000000000000000ull) {
      result.u = bits | ((tu_int)ARM64EC_X87_EXP_MAX << 112);
      return result.f;
    }
    // Propagate NaNs with the quiet bit set; an invalid encoding becomes the
    // indefinite NaN.
    if ((significand >> 63) == 0) {
      result.u = ((tu_int)1 << 127) | ((tu_int)ARM64EC_X87_EXP_MAX << 112) |
                 ((tu_int)1 << 111);
      return result.f;
    }
    result.u = bits | ((tu_int)ARM64EC_X87_EXP_MAX << 112) |
               ((tu_int)(significand & 0x7fffffffffffffffull) << 49) |
               ((tu_int)1 << 111);
    return result.f;
  }

  if (e != 0 && (significand >> 63) == 0) {
    // Invalid encoding: the indefinite NaN.
    result.u = ((tu_int)1 << 127) | ((tu_int)ARM64EC_X87_EXP_MAX << 112) |
               ((tu_int)1 << 111);
    return result.f;
  }

  if (e == 0) {
    if (significand == 0) {
      result.u = bits;
      return result.f;
    }
    if (significand >> 63) {
      // A pseudo-denormal is the normal number with exponent one.
      bits |= (tu_int)1 << 112;
      significand &= 0x7fffffffffffffffull;
    }
    // A denormal (or the pseudo-denormal above) maps onto a binary128
    // subnormal, which uses the same scale factor.
    result.u = bits | ((tu_int)significand << 49);
    return result.f;
  }

  // Normals share the exponent bias and the explicit integer bit.
  result.u = bits | ((tu_int)e << 112) |
             ((tu_int)(significand & 0x7fffffffffffffffull) << 49);
  return result.f;
}

/// Round a 113-bit significand (bit 112 implicit) with a 49-bit dropped part.
static uint64_t arm64ec_round_significand(tu_int significand) {
  tu_int kept = significand >> 49;
  tu_int dropped = significand & (((tu_int)1 << 49) - 1);

  if (dropped > ((tu_int)1 << 48) ||
      (dropped == ((tu_int)1 << 48) && (kept & 1)))
    ++kept;
  return (uint64_t)kept;
}

COMPILER_RT_ABI arm64ec_x87_reg_image __arm64ec_trunctfxf2(arm64ec_f128 a) {
  arm64ec_f128_bits src;
  uint64_t significand;
  unsigned sign, e;
  int exponent;

  src.f = a;
  sign = (unsigned)(src.u >> 127);
  e = (unsigned)((src.u >> 112) & 0x7fffu);

  if (e == ARM64EC_X87_EXP_MAX) {
    if (!(src.u & (((tu_int)1 << 112) - 1)))
      significand = 0x8000000000000000ull;
    else
      significand = (uint64_t)(((src.u & (((tu_int)1 << 112) - 1)) >> 48) |
                               0xc000000000000000ull);
    return ((arm64ec_x87_reg_image)((sign << 15) | ARM64EC_X87_EXP_MAX) << 64) |
           significand;
  }

  if (e == 0) {
    if ((src.u & (((tu_int)1 << 112) - 1)) == 0)
      return (arm64ec_x87_reg_image)sign << 79;
    // Every binary128 subnormal is an x87 denormal.
    significand = arm64ec_round_significand(src.u << 1);
    exponent = 0;
  } else {
    significand = arm64ec_round_significand(src.u | ((tu_int)1 << 112));
    exponent = (int)e;
    // Rounding may have carried into the next binade.
    if (significand == 0) {
      significand = 0x8000000000000000ull;
      ++exponent;
    }
  }

  if (exponent >= (int)ARM64EC_X87_EXP_MAX) {
    significand = 0x8000000000000000ull;
    exponent = ARM64EC_X87_EXP_MAX;
  }

  return ((arm64ec_x87_reg_image)((sign << 15) | (unsigned)exponent) << 64) |
         significand;
}
