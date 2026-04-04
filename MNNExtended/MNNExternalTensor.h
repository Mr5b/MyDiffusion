#ifndef MY_MODULE_MNN_EXTERNAL_TENSOR_H
#define MY_MODULE_MNN_EXTERNAL_TENSOR_H

#include <MNN/expr/Module.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/expr/MathOp.hpp>
#include "Scalar.h"
#include "dispatch.h"
#include "mymodule_assert.h"

/*MNN::Express::VARP _Const
(
    const void* ptr, INTS shape, Dimensionformat format, halide_type_t type
)
{
    Variable::Info info;
    info.dim = std::move(shape);
    info.order = format;
    info.type = type;
    return (Variable::create(Expr::create(std::move(info), ptr, VARP::CONSTANT)));
}*/
namespace DonNotKnowHowToNameIt
{

std::vector<MNN::Express::VARP> chunk(MNN::Express::VARP input, int chunks, int axis = 0)
{
    if (axis < 0) axis += input->getInfo()->dim.size();
    int dim_size = input->getInfo()->dim[axis];
    
    int block_size = dim_size / chunks;
    int remainder = dim_size % chunks;
    MNN::Express::INTS size_splits;
    int larger_block_size = block_size + 1;
    
    for (int i = 0; i < remainder; i++)
    {
        size_splits.push_back(larger_block_size);
    }
    for (int i = remainder; i < chunks; i++)
    {
        size_splits.push_back(block_size);
    }
    
    return _Split(input, size_splits, axis);
}

MNN::Express::VARP _ConstRef
(
    const void* ptr, MNN::Express::INTS shape,
    MNN::Express::Dimensionformat format = MNN::Express::Dimensionformat::NHWC,
    halide_type_t type = halide_type_of<float>()
)
{
    MNN::Express::Variable::Info info;
    info.dim = std::move(shape);
    info.order = format;
    info.type = type;
    return
    (
        MNN::Express::Variable::create
        (
            MNN::Express::Expr::create
            (
                std::move(info),
                ptr,
                MNN::Express::VARP::CONSTANT,
                MNN::Express::Expr::MemoryType::REF
            )
        )
    );
}


MNN::Express::VARP createScalar
(
    const MyTensor::Utils::Scalar& value,
    MNN::Express::Dimensionformat format = MNN::Express::Dimensionformat::NHWC,
    halide_type_t type = halide_type_of<float>()
)
{
    MNN::Express::VARP scalar_tensor;
    
    DISPATCH_ALL_TYPES
    (
        type,
        "",
        [&]
        {
            scalar_t temp = value.to<scalar_t>();
            scalar_tensor = MNN::Express::_Scalar(&temp, type);
        }
    );
    scalar_tensor.setOrder(format);
    
    return scalar_tensor;
}


MNN::Express::VARP fillValue
(
    const MyTensor::Utils::Scalar& value,
    MNN::Express::INTS shape = {},
    MNN::Express::Dimensionformat format = MNN::Express::Dimensionformat::NHWC,
    halide_type_t type = halide_type_of<float>()
)
{    
    MNN::Express::VARP dims_tensor =
        MNN::Express::_Const
        (
            shape.data(),
            {static_cast<int>(shape.size())},
            format,
            halide_type_of<int>()
        );
    
    MNN::Express::VARP scalar_tensor = createScalar(value, format, type);
    
    MNN::Express::VARP dst = MNN::Express::_Fill(dims_tensor, scalar_tensor);
    
    MY_ASSERT(dst->getInfo()->order == format, "dst");
    return dst;
}



void replacePtr(MNN::Express::VARP x, void* new_ptr)
{
    MNN::Express::Variable::Info info = *(x->getInfo());
    auto old_expr_pair = x->expr();
    MNN::Express::EXPRP new_expr =
        MNN::Express::Expr::create
        (
            std::move(info),
            new_ptr,
            old_expr_pair.first->inputType(),
            MNN::Express::Expr::MemoryType::REF
        );
    
    x->setExpr(new_expr, old_expr_pair.second);
}


MNN::Express::VARP _RandomNormal
(
    MNN::Express::VARP shape,
    halide_type_t type = halide_type_of<float>(),
    float mean = 0.0f,
    float stddev = 1.0f,
    int seed0 = 0,
    int seed1 = 0
)
{
    using namespace MNN::Express;
    
    VARP U1 = _RandomUnifom(shape, type, 0.0f, 1.0f, seed0, seed1);
    VARP U2 = _RandomUnifom(shape, type, 0.0f, 1.0f, seed0 + 12345, seed1 + 67890);
    
    MY_ASSERT(type.code == 2, "");
    
    VARP two_pi = createScalar(2.0f * M_PI, Dimensionformat::NCHW, type);
    VARP neg_two = createScalar(-2.0f, Dimensionformat::NCHW, type);
    VARP sqrt_arg = neg_two * _Log(U1);
    VARP Z0 = _Sqrt(sqrt_arg) * _Cos(two_pi * U2);
    
    VARP result = (Z0 * createScalar(stddev, NCHW, type)) + createScalar(mean, NCHW, type);
    return result;
}

MNN::Express::VARP _RandomNormal
(
    MNN::Express::INTS shape,
    halide_type_t type = halide_type_of<float>(),
    float mean = 0.0f,
    float stddev = 1.0f,
    int seed0 = 0,
    int seed1 = 0
)
{
    MNN::Express::VARP shape_tensor =
        _Const
        (
            shape.data(),
            {int(shape.size())},
            MNN::Express::Dimensionformat::NCHW,
            halide_type_of<int>()
        );
        
    return _RandomNormal(shape_tensor, type, mean, stddev, seed0, seed1);
}

}//namespace DonNotKnowHowToNameIt
#endif