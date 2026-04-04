#ifndef MY_MODULE_EMBEDDING
#define MY_MODULE_EMBEDDING

#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "MyModule.h"

namespace DonNotKnowHowToNameIt
{

using namespace MNN::Express;

class Embedding : public MyModule
{
public:
    Embedding
    (
        int num_embeddings, int embedding_dim,
        halide_type_t dtype = halide_type_of<float>()
    ) :
        weight_(fillValue(0, {num_embeddings, embedding_dim}, NCHW, dtype))
    {
        register_parameter("weight", weight_);
    }
    
    virtual std::vector<MNN::Express::VARP> onForward(const std::vector<MNN::Express::VARP>& inputs) override
    {
        weight_.fix(VARP::CONSTANT);
        VARP indices = inputs[0];

        VARP output = _Gather(weight_, indices);
        return {output};
    }
    
    VARP weight_;
};



}//namespace DonNotKnowHowToNameIt

#endif