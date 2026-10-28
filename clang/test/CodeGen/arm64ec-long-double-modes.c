// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-64 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LD64
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-80 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LD80
// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -mlong-double-128 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=LD128

// The three widths select the format of `long double`; only the 80-bit one is
// the software x87 format that goes through the ARM64EC runtime helpers.

long double g = 1.0L;
long double add(long double a, long double b) { return a + b; }
int cmp(long double a, long double b) { return a < b; }

// LD64: @g = {{.*}}global double 1.000000e+00
// LD64-LABEL: define {{.*}}double @add(double noundef %a, double noundef %b)
// LD64: fadd double
// LD64-LABEL: define {{.*}}i32 @cmp(double noundef %a, double noundef %b)
// LD64: fcmp olt double
// LD64-NOT: __arm64ec_

// LD80: @g = {{.*}}global i128 302222231531620438900736, align 16
// LD80-LABEL: define {{.*}}i128 @add(i128 noundef %a, i128 noundef %b)
// LD80: call double @__arm64ec_load_x87_long_double(
// LD80-LABEL: define {{.*}}i32 @cmp(i128 noundef %a, i128 noundef %b)
// LD80: icmp ult i80

// LD128: @g = {{.*}}global fp128
// LD128-LABEL: define {{.*}}fp128 @add(fp128 noundef %a, fp128 noundef %b)
// LD128-LABEL: define {{.*}}i32 @cmp(fp128 noundef %a, fp128 noundef %b)
// LD128-NOT: __arm64ec_
