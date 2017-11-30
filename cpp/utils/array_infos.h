#ifndef TFKERNELS_ARRAY_INFOS_IMPL_H
#define TFKERNELS_ARRAY_INFOS_IMPL_H

#include "cuda_includes_minimal.h"
#include "assert.h"
#include <stddef.h>
#include <initializer_list>
#include <stdio.h>

#include "core_types.h"
#include "error_handling_cpu.h"




// #define DO_BOUNDS_CHECK
#ifdef DO_BOUNDS_CHECK

#ifdef USING_NVCC

#define RUN_BOUNDS_CHECK_0D
#define RUN_BOUNDS_CHECK_1D RUN_BOUNDS_CHECK_0D assert(i < dims[0]);
#define RUN_BOUNDS_CHECK_2D RUN_BOUNDS_CHECK_1D assert(j < dims[1]);
#define RUN_BOUNDS_CHECK_3D RUN_BOUNDS_CHECK_2D assert(k < dims[2]);
#define RUN_BOUNDS_CHECK_4D RUN_BOUNDS_CHECK_3D assert(l < dims[3]);
#define RUN_JAGGED_BOUNDS_CHECK if (idx >= num_arrays) printf("Only %d arrays in jagged array. Requested %d.\n", num_arrays, idx );

#else

#include "error_handling_cpu.h"
#include <iostream>

#define RUN_BOUNDS_CHECK_0D
#define RUN_BOUNDS_CHECK_1D RUN_BOUNDS_CHECK_0D assert_abort_with_message(i < dims[0]);
#define RUN_BOUNDS_CHECK_2D RUN_BOUNDS_CHECK_1D assert_abort_with_message(j < dims[1]);
#define RUN_BOUNDS_CHECK_3D RUN_BOUNDS_CHECK_2D assert_abort_with_message(k < dims[2]);
#define RUN_BOUNDS_CHECK_4D RUN_BOUNDS_CHECK_3D assert_abort_with_message(l < dims[3]);
#define RUN_JAGGED_BOUNDS_CHECK if (idx >= num_arrays) printf("Only %d arrays in jagged array. Requested %d.\n", num_arrays, idx );


#endif

#endif

#ifndef DO_BOUNDS_CHECK
#define RUN_BOUNDS_CHECK_1D
#define RUN_BOUNDS_CHECK_2D
#define RUN_BOUNDS_CHECK_3D
#define RUN_BOUNDS_CHECK_4D
#define RUN_JAGGED_BOUNDS_CHECK
#endif

// Tensorflow uses row major (C style indexing) tensors.
// Both row and col (F) major references are here for reference purposes.

// in all of these indexing operations, len_ijk can be either the length in that dimension of the array
// or if the array is actually a sub array of a larger one, then the length in that dimension of the
// whole larger array

// For the ArrayInfo objects, when interfacing with Julia code, to be binary compatible, we ensured they are all PODs.
// ie we do not add any constructors, so that the binary layout is guaranteed to be a certain way.
// For c++ only code, including cuda, we can have constructors.


#define Idx2dC (i * len_j + j)
#define Idx3dC (i * len_j * len_k + j * len_k + k)
#define Idx4dC (i * len_j * len_k * len_l + j * len_k * len_l + k * len_l + l)

#define Idx2dF (j * len_i + i)
#define Idx3dF (k * len_i * len_j + j * len_i + i)
#define Idx4dF (l * len_i * len_j * len_k + k * len_i * len_j + j * len_i + i)

namespace lng {




  template<typename T>
  __device__ inline
  T & ref_2d_c(
      T *arr,
      IntSys i, IntSys j,
      IntSys len_i, IntSys len_j
  ) {
    return arr[Idx2dC];
  }

  template<typename T>
  __device__ inline
  const T & ref_2d_c(
      const T *arr,
      IntSys i, IntSys j,
      IntSys len_i, IntSys len_j
  ) {
    return arr[Idx2dC];
  }

  template<typename T>
  __device__ inline
  T & ref_3d_c(
      T *arr,
      IntSys i, IntSys j, IntSys k,
      IntSys len_i, IntSys len_j, IntSys len_k
  ) {
    return arr[Idx3dC];
  }

  template<typename T>
  __device__ inline
  const T &ref_3d_c(
      const T *arr,
      IntSys i, IntSys j, IntSys k,
      IntSys len_i, IntSys len_j, IntSys len_k
  ) {
    return arr[Idx3dC];
  }

  template<typename T>
  __device__ inline
  T & ref_4d_c(
      T *arr,
      IntSys i, IntSys j, IntSys k, IntSys l,
      IntSys len_i, IntSys len_j, IntSys len_k, IntSys len_l
  ) {
    return arr[Idx4dC];
  }

  template<typename T>
  __device__ inline
  const T &ref_4d_c(
      const T *arr,
      IntSys i, IntSys j, IntSys k, IntSys l,
      IntSys len_i, IntSys len_j, IntSys len_k, IntSys len_l
  ) {
    return arr[Idx4dC];
  }















  template<typename T>
  __device__ inline
  T & ref_2d_f(
      T *arr,
      IntSys i, IntSys j,
      IntSys len_i, IntSys len_j
  ) {
    return arr[Idx2dF];
  }

  template<typename T>
  __device__ inline
  const T & ref_2d_f(
      const T *arr,
      IntSys i, IntSys j,
      IntSys len_i, IntSys len_j
  ) {
    return arr[Idx2dF];
  }

  template<typename T>
  __device__ inline
  T &ref_3d_f(
      T *arr,
      IntSys i, IntSys j, IntSys k,
      IntSys len_i, IntSys len_j, IntSys len_k
  ) {
    return arr[Idx3dF];
  }

  template<typename T>
  __device__ inline
  const T &ref_3d_f(
      const T *arr,
      IntSys i, IntSys j, IntSys k,
      IntSys len_i, IntSys len_j, IntSys len_k
  ) {
    return arr[Idx3dF];
  }

  template<typename T>
  __device__ inline
  T &ref_4d_f(
      T *arr,
      IntSys i, IntSys j, IntSys k, IntSys l,
      IntSys len_i, IntSys len_j, IntSys len_k, IntSys len_l
  ) {
    return arr[Idx4dF];
  }

  template<typename T>
  __device__ inline
  const T &ref_4d_f(
      const T *arr,
      IntSys i, IntSys j, IntSys k, IntSys l,
      IntSys len_i, IntSys len_j, IntSys len_k, IntSys len_l
  ) {
    return arr[Idx4dF];
  }






  /**
   * Strided arrays.
   * Row major.
   */
  template<typename T, IntSys NDIMS>
  class ArrayInfoNdStrided {
  private:
    T * const _ptr;
  public:
    IntSys dims[NDIMS];
    IntSys actual_dims[NDIMS];

    ArrayInfoNdStrided(): _ptr(NULL) {}

    /**
     *
     * @param ptr
     * @param dims              The size of this array along each dimension.
     * @param actual_dims       This array is based on a larger array. The dimensions of the larger array.
     */
    __device__
    ArrayInfoNdStrided(T* ptr, IntSys dims[NDIMS], IntSys actual_dims[NDIMS]):
        _ptr(ptr){
      for (IntSys i = 0; i < NDIMS; i++) {
        this->dims[i] = dims[i];
        this->actual_dims[i] = actual_dims[i];
        assert(actual_dims[i] >= dims[i]); // TODO require
      }
    }

    const T * ptr() const { return _ptr; }
    T * ptr() { return _ptr; }



    __device__ inline
    T &operator()(IntSys i) {
      static_assert(NDIMS == 1, "NDIMS != 1");
      RUN_BOUNDS_CHECK_1D
      return _ptr[i];
    }
    __device__ inline
    const T &operator()(IntSys i) const {
      static_assert(NDIMS == 1, "NDIMS != 1");
      RUN_BOUNDS_CHECK_1D
      return _ptr[i];
    }

    __device__ inline
    T &operator()(IntSys i, IntSys j) {
      static_assert(NDIMS == 2, "NDIMS != 2");
      RUN_BOUNDS_CHECK_2D
      return ref_2d_c(_ptr, i, j, actual_dims[0], actual_dims[1]);
    }
    __device__ inline
    const T &operator()(IntSys i, IntSys j) const {
      static_assert(NDIMS == 2, "NDIMS != 2");
      RUN_BOUNDS_CHECK_2D
      return ref_2d_c(_ptr, i, j, actual_dims[0], actual_dims[1]);
    }

    __device__ inline
    T &operator()(IntSys i, IntSys j, IntSys k) {
      static_assert(NDIMS == 3, "NDIMS != 3");
      RUN_BOUNDS_CHECK_3D
      return ref_3d_c(_ptr, i, j, k, actual_dims[0], actual_dims[1], actual_dims[2]);
    }
    __device__ inline
    const T &operator()(IntSys i, IntSys j, IntSys k) const {
      static_assert(NDIMS == 3, "NDIMS != 3");
      RUN_BOUNDS_CHECK_3D
      return ref_3d_c(_ptr, i, j, k, actual_dims[0], actual_dims[1], actual_dims[2]);
    }

    __device__ inline
    T &operator()(IntSys i, IntSys j, IntSys k, IntSys l) {
      static_assert(NDIMS == 4, "NDIMS != 4");
      RUN_BOUNDS_CHECK_4D
      return ref_4d_c(_ptr, i, j, k, l, actual_dims[0], actual_dims[1], actual_dims[2], actual_dims[3]);
    }
    __device__ inline
    const T &operator()(IntSys i, IntSys j, IntSys k, IntSys l) const {
      static_assert(NDIMS == 4, "NDIMS != 4");
      RUN_BOUNDS_CHECK_4D
      return ref_4d_c(_ptr, i, j, k, l, actual_dims[0], actual_dims[1], actual_dims[2], actual_dims[3]);
    }

    __device__ inline IntSys dim0() const { static_assert(NDIMS >= 1, "NDIMS < 1"); return dims[0]; }
    __device__ inline IntSys dim1() const { static_assert(NDIMS >= 2, "NDIMS < 2"); return dims[1]; }
    __device__ inline IntSys dim2() const { static_assert(NDIMS >= 3, "NDIMS < 3"); return dims[2]; }
    __device__ inline IntSys dim3() const { static_assert(NDIMS >= 4, "NDIMS < 4"); return dims[3]; }


  };






  /**
   * A simple struct which doesn't own the underlying memory it points to.
   */
  template<typename T, IntSys NDIMS>
  class ArrayInfoNd {
  private:
    T * _ptr;
  public:
    IntSys dims[NDIMS];

    ArrayInfoNd(): _ptr(NULL) {}

    __device__
    ArrayInfoNd(T* ptr, IntSys dims[NDIMS]):
        _ptr(ptr){
      for (IntSys i = 0; i < NDIMS; i++) {
        this->dims[i] = dims[i];
      }
    }

    __device__ const T * ptr() const { return _ptr; }
    __device__ T * ptr() { return _ptr; }


    // accessors for 0-4 dimensions
    // could eventually create more if needed.
    // Access should be very fast because inline and dims is fixed size at compile time.
    // Bounds checking usually disabled, unless macro defined.

    __device__ inline
    T &operator()() {
      static_assert(NDIMS == 0, "NDIMS != 0");
      return _ptr[0];
    }
    __device__ inline
    const T &operator()() const {
      static_assert(NDIMS == 0, "NDIMS != 0");
      return _ptr[0];
    }

    __device__ inline
    T &operator()(IntSys i) {
      static_assert(NDIMS == 1, "NDIMS != 1");
      RUN_BOUNDS_CHECK_1D
      return _ptr[i];
    }
    __device__ inline
    const T &operator()(IntSys i) const {
      static_assert(NDIMS == 1, "NDIMS != 1");
      RUN_BOUNDS_CHECK_1D
      return _ptr[i];
    }

    __device__ inline
    T &operator()(IntSys i, IntSys j) {
      static_assert(NDIMS == 2, "NDIMS != 2");
      RUN_BOUNDS_CHECK_2D
      return ref_2d_c(_ptr, i, j, dims[0], dims[1]);
    }
    __device__ inline
    const T &operator()(IntSys i, IntSys j) const {
      static_assert(NDIMS == 2, "NDIMS != 2");
      RUN_BOUNDS_CHECK_2D
      return ref_2d_c(_ptr, i, j, dims[0], dims[1]);
    }

    __device__ inline
    T &operator()(IntSys i, IntSys j, IntSys k) {
      static_assert(NDIMS == 3, "NDIMS != 3");
      RUN_BOUNDS_CHECK_3D
      return ref_3d_c(_ptr, i, j, k, dims[0], dims[1], dims[2]);
    }
    __device__ inline
    const T &operator()(IntSys i, IntSys j, IntSys k) const {
      static_assert(NDIMS == 3, "NDIMS != 3");
      RUN_BOUNDS_CHECK_3D
      return ref_3d_c(_ptr, i, j, k, dims[0], dims[1], dims[2]);
    }

    __device__ inline
    T &operator()(IntSys i, IntSys j, IntSys k, IntSys l) {
      static_assert(NDIMS == 4, "NDIMS != 4");
      RUN_BOUNDS_CHECK_4D
      return ref_4d_c(_ptr, i, j, k, l, dims[0], dims[1], dims[2], dims[3]);
    }
    __device__ inline
    const T &operator()(IntSys i, IntSys j, IntSys k, IntSys l) const {
      static_assert(NDIMS == 4, "NDIMS != 4");
      RUN_BOUNDS_CHECK_4D
      return ref_4d_c(_ptr, i, j, k, l, dims[0], dims[1], dims[2], dims[3]);
    }

    __device__ inline IntSys dim0() const { static_assert(NDIMS >= 1, "NDIMS < 1"); return dims[0]; }
    __device__ inline IntSys dim1() const { static_assert(NDIMS >= 2, "NDIMS < 2"); return dims[1]; }
    __device__ inline IntSys dim2() const { static_assert(NDIMS >= 3, "NDIMS < 3"); return dims[2]; }
    __device__ inline IntSys dim3() const { static_assert(NDIMS >= 4, "NDIMS < 4"); return dims[3]; }


    /**
     * Takes the slice of the array, eg arr[idx, 0:h, 0:w]
     *   where leading_index = idx
     *   and trailing_dims = [h, w]
     *
     * Only makes sense for a row major array.
     *
     * @param leading_index
     * @param trailing_dims
     * @return
     */
    __device__
    const ArrayInfoNdStrided<T, NDIMS - 1> subarray(IntSys leading_index, std::initializer_list<IntSys> trailing_dims) const {


      assert(trailing_dims.size() == NDIMS - 1);
      IntSys trailing_dims_arr[NDIMS - 1];
      for (IntSys i = 0; i < NDIMS - 1; i++) {
        trailing_dims_arr[i] = *(trailing_dims.begin() + i);
      }

      assert(leading_index < dim0());

      // find pointer to start of slice
      IntSys leading_stride = 1;
      for (IntSys i = 1; i < NDIMS; i++) {
        leading_stride *= dims[i];
      }
      T* p_slice = _ptr + leading_index * leading_stride;

      // set up actual dims
      IntSys actual_dims[NDIMS - 1];
      for (IntSys i = 0; i < NDIMS - 1; i++) {
        assert(trailing_dims_arr[i] <= dims[i+1]);
        actual_dims[i] = dims[i+1];
      }

      return ArrayInfoNdStrided<T, NDIMS - 1>(p_slice, trailing_dims_arr, actual_dims);
    };

    __device__
    ArrayInfoNdStrided<T, NDIMS - 1> subarray(IntSys leading_index, std::initializer_list<IntSys> trailing_dims) {
      return static_cast<const ArrayInfoNd<T, NDIMS> *>(this)->subarray(leading_index, trailing_dims);
    }

    __device__
    void fill(T val) {
      for (IntSys i = 0; i < underlying_length(); i++) {
        _ptr[i] = val;
      }
    }

    __device__
    IntSys underlying_length() const {
      IntSys prod = 1;
      for (IntSys i = 0; i < NDIMS; i++) {
        prod *= dims[i];
      }
      return prod;
    }




  };


  // allows to create const array
  template<typename T, IntSys NDIMS>
  __device__ const ArrayInfoNd<T, NDIMS> create_array_info_nd(const T* ptr, IntSys dims[NDIMS]) {
    return ArrayInfoNd<T, NDIMS>(const_cast<T*>(ptr), dims);
  }
  template<typename T, IntSys NDIMS>
  __device__ ArrayInfoNd<T, NDIMS> create_array_info_nd(T* ptr, IntSys dims[NDIMS]) {
    return ArrayInfoNd<T, NDIMS>(ptr, dims);
  }


  typedef ArrayInfoNd<IntSys, 1> Arr1Int;
  typedef ArrayInfoNd<IntSys, 2> Arr2Int;
  typedef ArrayInfoNd<IntSys, 3> Arr3Int;
  typedef ArrayInfoNd<IntSys, 4> Arr4Int;
  typedef ArrayInfoNd<FloatSys, 1> Arr1Flt;
  typedef ArrayInfoNd<FloatSys, 2> Arr2Flt;
  typedef ArrayInfoNd<FloatSys, 3> Arr3Flt;
  typedef ArrayInfoNd<FloatSys, 4> Arr4Flt;

  template<typename T> using Arr1 = ArrayInfoNd<T, 1>;
  template<typename T> using Arr2 = ArrayInfoNd<T, 2>;
  template<typename T> using Arr3 = ArrayInfoNd<T, 3>;
  template<typename T> using Arr4 = ArrayInfoNd<T, 4>;













  /**
   * Jagged array, or array of arrays
   * @tparam T          type contained in array
   * @tparam NDIMS      number of dims of each contained array
   */
  template<typename T, int NDIMS>
  class Arrays {
  private:
    ArrayInfoNd<T, 1> underlying_buffer;  // N x 1
    const Arr2Int shapes;                 // NDIMS x N
    const Arr1Int offsets;                // N x 1

  public:
    const IntSys length;

    /**
     *
     * @param length                number of arrays N
     * @param underlying_buffer
     * @param offsets               offsets into the underlying buffer for the start of each array
     * @param shapes                NDIMS x N   shapes of all arrays
     */
    __device__
    Arrays(IntSys length, ArrayInfoNd<T, 1> & underlying_buffer, const Arr2Int & shapes, const Arr1Int & offsets):
        length(length), underlying_buffer(underlying_buffer), shapes(shapes), offsets(offsets) {

      // TODO require
      assert(shapes.dim0() == NDIMS);
      assert(shapes.dim1() == length);
      assert(offsets.dim0() == length);
    }

    __device__
    const ArrayInfoNd<T, NDIMS> operator[](IntSys idx) const {
      assert(idx < length);  // TODO require

      IntSys dims[NDIMS];
      for (IntSys i = 0; i < NDIMS; i++) {
        dims[i] = shapes(i, idx);
      }

      return create_array_info_nd<T, NDIMS>(underlying_buffer.ptr() + offsets(idx), dims);
    }

    __device__
    ArrayInfoNd<T, NDIMS> operator[](IntSys idx) {
      return static_cast<const Arrays<T, NDIMS> *>(this)->operator[](idx);
    }

  };


  typedef Arrays<IntSys, 1> Arrays1dInt;
  typedef Arrays<IntSys, 2> Arrays2dInt;
  typedef Arrays<IntSys, 3> Arrays3dInt;
  typedef Arrays<IntSys, 4> Arrays4dInt;

  typedef Arrays<FloatSys, 1> Arrays1dFlt;
  typedef Arrays<FloatSys, 2> Arrays2dFlt;
  typedef Arrays<FloatSys, 3> Arrays3dFlt;
  typedef Arrays<FloatSys, 4> Arrays4dFlt;

  template<typename T> using Arrs1 = Arrays<T, 1>;
  template<typename T> using Arrs2 = Arrays<T, 2>;
  template<typename T> using Arrs3 = Arrays<T, 3>;
  template<typename T> using Arrs4 = Arrays<T, 4>;













}



#endif //TFKERNELS_ARRAY_INFOS_IMPL_H
