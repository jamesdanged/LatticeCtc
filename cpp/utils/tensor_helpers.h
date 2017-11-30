#ifndef TFKERNELS_TENSOR_HELPERS_H
#define TFKERNELS_TENSOR_HELPERS_H

#include "tensorflow/core/framework/tensor.h"
#include "array_infos.h"



namespace lng {

  using namespace tensorflow;



  template<typename T, IntSys N> ArrayInfoNd<T, N> to_info(Tensor & t) {
    auto shape = t.shape();
    if (N == 0) {
      IntSys dims[0];
      return ArrayInfoNd<T, N>(t.flat<T>().data(), dims);
    }
    if (N == 1) {
      IntSys dims[1];
      dims[0] = (IntSys)shape.dim_size(0);
      return ArrayInfoNd<T, N>(t.flat<T>().data(), dims);
    }
    if (N == 2) {
      IntSys dims[2];
      dims[0] = (IntSys)shape.dim_size(0);
      dims[1] = (IntSys)shape.dim_size(1);
      return ArrayInfoNd<T, N>(t.flat<T>().data(), dims);
    }
    if (N == 3) {
      IntSys dims[3];
      dims[0] = (IntSys)shape.dim_size(0);
      dims[1] = (IntSys)shape.dim_size(1);
      dims[2] = (IntSys)shape.dim_size(2);
      return ArrayInfoNd<T, N>(t.flat<T>().data(), dims);
    }
    if (N == 4) {
      IntSys dims[4];
      dims[0] = (IntSys)shape.dim_size(0);
      dims[1] = (IntSys)shape.dim_size(1);
      dims[2] = (IntSys)shape.dim_size(2);
      dims[3] = (IntSys)shape.dim_size(3);
      return ArrayInfoNd<T, N>(t.flat<T>().data(), dims);
    }
    abort();
  };

  template<typename T, IntSys N> const ArrayInfoNd<T, N> to_info(const Tensor & t) {
    return to_info<T, N>(const_cast<Tensor &>(t));
  }


  template<typename T> DataType get_dt() { abort(); }
  template<> inline DataType get_dt<float>() { return DT_FLOAT; }
  template<> inline DataType get_dt<double>() { return DT_DOUBLE; }
  template<> inline DataType get_dt<int32>() { return DT_INT32; }
  template<> inline DataType get_dt<int64>() { return DT_INT64; }


  /**
   * May own a tensor or simply reference a tensor by pointer.
   */
  template<typename T, IntSys NDIMS>
  class TensorWrapper {
  private:
    Tensor contained;
    Tensor * referenced = NULL;
    bool is_pointer = false;
    ArrayInfoNd<T, NDIMS> _info;
  public:


    TensorWrapper(): contained(), referenced(NULL), is_pointer(false), _info() {}
    TensorWrapper(Tensor tensor): contained(tensor), _info(to_info<T, NDIMS>(tensor)), is_pointer(false) {}
    // TensorWrapper(const Tensor & tensor): contained(tensor), _info(to_info<T, NDIMS>(tensor)) {}
    TensorWrapper(Tensor * p_tensor): referenced(p_tensor), _info(to_info<T, NDIMS>(*p_tensor)), is_pointer(true) {}

    Tensor & tensor() {
      if (is_pointer) {
        return *referenced;
      } else {
        return contained;
      }
    }

    __device__ inline T &operator()() { static_assert(NDIMS == 0, "NDIMS != 0");  return _info(); }
    __device__ inline const T &operator()() const { static_assert(NDIMS == 0, "NDIMS != 0"); return _info(); }
    __device__ inline T &operator()(IntSys i) { static_assert(NDIMS == 1, "NDIMS != 1");  return _info(i); }
    __device__ inline const T &operator()(IntSys i) const { static_assert(NDIMS == 1, "NDIMS != 1"); return _info(i); }
    __device__ inline T &operator()(IntSys i, IntSys j) { static_assert(NDIMS == 2, "NDIMS != 2"); return _info(i, j); }
    __device__ inline const T &operator()(IntSys i, IntSys j) const { static_assert(NDIMS == 2, "NDIMS != 2"); return _info(i, j); }
    __device__ inline T &operator()(IntSys i, IntSys j, IntSys k) { static_assert(NDIMS == 3, "NDIMS != 3"); return _info(i, j, k); }
    __device__ inline const T &operator()(IntSys i, IntSys j, IntSys k) const { static_assert(NDIMS == 3, "NDIMS != 3"); return _info(i, j, k); }
    __device__ inline T &operator()(IntSys i, IntSys j, IntSys k, IntSys l) { static_assert(NDIMS == 4, "NDIMS != 4"); return _info(i, j, k, l); }
    __device__ inline const T &operator()(IntSys i, IntSys j, IntSys k, IntSys l) const { static_assert(NDIMS == 4, "NDIMS != 4"); return _info(i, j, k, l); }

    ArrayInfoNd<T, NDIMS> & info() { return _info; }
    const ArrayInfoNd<T, NDIMS> & info() const { return _info; }

//    // implicit conversions
//    operator ArrayInfoNd<T, NDIMS> & () { return _info; }
//    operator const ArrayInfoNd<T, NDIMS> & () const { return _info; }


    IntSys ndims() const { return NDIMS; }
    IntSys dim(IntSys d) const {
      assert(d < NDIMS);
      return _info.dims[d];
    }

    IntSys dim0() const { static_assert(NDIMS >= 1, "NDIMS < 1"); return _info.dims[0]; }
    IntSys dim1() const { static_assert(NDIMS >= 2, "NDIMS < 2"); return _info.dims[1]; }
    IntSys dim2() const { static_assert(NDIMS >= 3, "NDIMS < 3"); return _info.dims[2]; }
    IntSys dim3() const { static_assert(NDIMS >= 4, "NDIMS < 4"); return _info.dims[3]; }

  };






  string shape_to_string(const Tensor & t);

  IntSys tensor_total_length(const Tensor & t);

  bool copy_d_to_h(const Eigen::GpuDevice & d, const void * src, void * dest, size_t byte_length);
}
#endif //TFKERNELS_TENSOR_HELPERS_H
