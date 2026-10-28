// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_x87_long_double

// ARM64EC keeps `long double` in the x87 80-bit format as an integer image.
// `__arm64ec_load_x87_long_double()` and `__arm64ec_store_x87_long_double()`
// are the only bridge to binary64, and the compiler routes every computation
// through them, so they are exercised here directly on explicit images instead
// of through the C `long double` type.

#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint64_t significand;
  uint16_t sign_exponent;
  uint16_t padding[3];
} x87_image;

double __arm64ec_load_x87_long_double(const x87_image *image);
void __arm64ec_store_x87_long_double(x87_image *image, double value);

#define EXP_MASK 0x7ff0000000000000ull
#define FRAC_MASK 0x000fffffffffffffull
#define SIGN_MASK 0x8000000000000000ull

#define INT_BIT 0x8000000000000000ull
#define X87_EXP_BIAS 16383
#define X87_EXP_MAX 0x7fffu

static int failures;

static uint64_t double_bits(double d) {
  union {
    double d;
    uint64_t u;
  } v;
  v.d = d;
  return v.u;
}

static int is_nan(double d) {
  uint64_t b = double_bits(d);
  return (b & EXP_MASK) == EXP_MASK && (b & FRAC_MASK) != 0;
}

static x87_image make_image(unsigned sign, unsigned biased_exponent,
                            uint64_t significand) {
  x87_image image;
  image.significand = significand;
  image.sign_exponent =
      (uint16_t)((sign << 15) | (biased_exponent & X87_EXP_MAX));
  image.padding[0] = image.padding[1] = image.padding[2] = 0;
  return image;
}

static void check_load_exact(const char *what, x87_image image,
                             uint64_t expected) {
  double result = __arm64ec_load_x87_long_double(&image);
  if (double_bits(result) != expected) {
    printf("load %s: expected %016llx, got %016llx\n", what,
           (unsigned long long)expected, (unsigned long long)double_bits(result));
    ++failures;
  }
}

static void check_load_nan(const char *what, x87_image image) {
  double result = __arm64ec_load_x87_long_double(&image);
  if (!is_nan(result)) {
    printf("load %s: expected a NaN, got %016llx\n", what,
           (unsigned long long)double_bits(result));
    ++failures;
  }
}

static void check_store(const char *what, double value, unsigned sign,
                        unsigned biased_exponent, uint64_t significand) {
  x87_image image;
  image.significand = ~0ull;
  image.sign_exponent = 0xffffu;
  image.padding[0] = image.padding[1] = image.padding[2] = 0xffffu;
  __arm64ec_store_x87_long_double(&image, value);
  if (image.significand != significand ||
      image.sign_exponent !=
          (uint16_t)((sign << 15) | (biased_exponent & X87_EXP_MAX))) {
    printf("store %s: expected %04x:%016llx, got %04x:%016llx\n", what,
           (unsigned)((sign << 15) | (biased_exponent & X87_EXP_MAX)),
           (unsigned long long)significand, (unsigned)image.sign_exponent,
           (unsigned long long)image.significand);
    ++failures;
  }
}

int main(void) {
  // Exact normals.
  check_load_exact("+1.0", make_image(0, X87_EXP_BIAS, INT_BIT),
                   0x3ff0000000000000ull);
  check_load_exact("+1.5", make_image(0, X87_EXP_BIAS, 0xc000000000000000ull),
                   0x3ff8000000000000ull);
  check_load_exact("-1.5", make_image(1, X87_EXP_BIAS, 0xc000000000000000ull),
                   0xbff8000000000000ull);

  // Both zeroes keep their sign.
  check_load_exact("+0.0", make_image(0, 0, 0), 0x0000000000000000ull);
  check_load_exact("-0.0", make_image(1, 0, 0), SIGN_MASK);

  // Infinities.
  check_load_exact("+inf", make_image(0, X87_EXP_MAX, INT_BIT), EXP_MASK);
  check_load_exact("-inf", make_image(1, X87_EXP_MAX, INT_BIT),
                   SIGN_MASK | EXP_MASK);

  // NaNs, including the x87 "real indefinite" and the pseudo-encodings whose
  // integer bit is clear.
  check_load_nan("qNaN", make_image(0, X87_EXP_MAX, 0xc000000000000000ull));
  check_load_nan("sNaN", make_image(0, X87_EXP_MAX, 0xa000000000000000ull));
  check_load_nan("indefinite", make_image(1, X87_EXP_MAX, INT_BIT | 0x1));
  check_load_nan("pseudo-NaN", make_image(0, X87_EXP_MAX, 0));

  // Overflow and underflow.
  check_load_exact("LDBL_MAX", make_image(0, 0x7ffe, ~0ull), EXP_MASK);
  check_load_exact("-LDBL_MAX", make_image(1, 0x7ffe, ~0ull),
                   SIGN_MASK | EXP_MASK);
  check_load_exact("smallest normal", make_image(0, 1, INT_BIT),
                   0x0000000000000000ull);
  check_load_exact("largest subnormal", make_image(0, 0, ~0ull),
                   0x0000000000000000ull);

  // Rounding: the significand has 64 bits, binary64 only 53.  The halfway
  // cases round to even and the ones just below round down.
  check_load_exact("1+2^-53", make_image(0, X87_EXP_BIAS, INT_BIT | 0x400),
                   0x3ff0000000000000ull);
  check_load_exact("1+2^-53+2^-63",
                   make_image(0, X87_EXP_BIAS, INT_BIT | 0x401),
                   0x3ff0000000000001ull);
  check_load_exact("1+3*2^-53", make_image(0, X87_EXP_BIAS, INT_BIT | 0xc00),
                   0x3ff0000000000002ull);
  check_load_exact("1+2^-54", make_image(0, X87_EXP_BIAS, INT_BIT | 0x200),
                   0x3ff0000000000000ull);

  // The padding bytes above the image do not take part in the conversion.
  {
    x87_image image = make_image(0, X87_EXP_BIAS, 0xc000000000000000ull);
    image.padding[0] = image.padding[1] = image.padding[2] = 0xffffu;
    check_load_exact("padding ignored", image, 0x3ff8000000000000ull);
  }

  // Storing binary64 produces the exact image, with the padding zeroed.
  check_store("+0.0", 0.0, 0, 0, 0);
  check_store("-0.0", -0.0, 1, 0, 0);
  check_store("+1.0", 1.0, 0, X87_EXP_BIAS, INT_BIT);
  check_store("+1.5", 1.5, 0, X87_EXP_BIAS, 0xc000000000000000ull);
  check_store("-1.5", -1.5, 1, X87_EXP_BIAS, 0xc000000000000000ull);
  check_store("+inf", __builtin_inf(), 0, X87_EXP_MAX, INT_BIT);
  check_store("-inf", -__builtin_inf(), 1, X87_EXP_MAX, INT_BIT);
  // The smallest binary64 value is still a normal number in the x87 format.
  check_store("smallest double", 0x1p-1074, 0, X87_EXP_BIAS - 1074, INT_BIT);

  // Round-tripping a binary64 value through the image is the identity, apart
  // from the payload of a NaN.
  {
    static const double values[] = {0.0,       -0.0,     1.0,     -1.0,
                                    1.5,       -1.5,     0x1p-1074,
                                    0x1p1023,  -0x1p1023};
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
      x87_image image;
      __arm64ec_store_x87_long_double(&image, values[i]);
      if (double_bits(__arm64ec_load_x87_long_double(&image)) !=
          double_bits(values[i])) {
        printf("round trip %u failed\n", i);
        ++failures;
      }
    }
  }

  if (failures) {
    printf("%d failure(s)\n", failures);
    return 1;
  }
  return 0;
}
