from cffi import FFI
ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef("""

    typedef float _Complex float_complex;
    typedef double _Complex double_complex;

    long double factorial(int n);

    void getSHreal(/* Input Arguments */
               int order,
               float* dirs_rad,
               int nDirs,
               /* Output Arguments */
               float* Y);

    void getSHcomplex(/* Input Arguments */
                      int order,
                      float* dirs_rad,
                      int nDirs,
                      /* Output Arguments */
                      float_complex* Y);

""")

# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source("_safpy",
"""
    #define SAF_USE_OPEN_BLAS_AND_LAPACKE
    #include "../../Spatial_Audio_Framework/framework/include/saf.h"   // the C header of the library
""",
     libraries=['../../Spatial_Audio_Framework/build/framework/saf', # library name, for the linker
                'lapacke'])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
