#ifndef MYDIFFUSION_LORA
#define MYDIFFUSION_LORA

#include <string>
#include <regex>
#include <unordered_map>


std::string stripWeightSuffix(const std::string& str)
{
    static std::regex weight_regex(R"((.+)\.weight$)");
    std::smatch m;
    if (std::regex_match(str, m, weight_regex))
    {
        return m[1].str();
    }
    return "";
}

std::string convertSD15KeyToLoRABase(const std::string& sd_key)
{
    using namespace std;

    if (sd_key.find("cond_stage_model.transformer.text_model.") == 0)
    {
        string base = stripWeightSuffix(sd_key);
        if (base.empty()) return "";

        base.replace
        (
            0, string("cond_stage_model.transformer.text_model.").length(),
            "lora_te_text_model_"
        );
                     
        for (char& c : base) if (c == '.') c = '_';
        return base;
    }

    if (sd_key.find("model.diffusion_model.") != 0)
    {
        return "";
    }

    string stripped = stripWeightSuffix(sd_key);
    if (stripped.empty()) return "";

    static regex unet_regex
    (
        R"(model\.diffusion_model\.(input_blocks|middle_block|output_blocks)\.(\d+)(?:\.(\d+))?\.(.+))"
    );
    smatch match;
    if (!regex_match(stripped, match, unet_regex))
    {
        return "";
    }

    string block_type = match[1].str();
    int block_idx = stoi(match[2].str());
    string sub_idx_str = match[3].str();
    string rest = match[4].str();

    string lora_prefix;
    int lora_block = -1, lora_attn = -1;

    if (block_type == "middle_block")
    {
        if (sub_idx_str == "1")
        {
            lora_prefix = "lora_unet_mid_block_attentions_0_";
        }
        else
        {
            return "";
        }
    } else if (block_type == "input_blocks" || block_type == "output_blocks")
    {
        static const unordered_map<int, pair<int, int>> input_map =
        {
            {1,  {0, 0}}, {2,  {0, 1}},
            {4,  {1, 0}}, {5,  {1, 1}},
            {7,  {2, 0}}, {8,  {2, 1}},
            {10, {3, 0}}, {11, {3, 1}}
        };
        static const unordered_map<int, pair<int, int>> output_map =
        {
            {0,  {0, 0}}, {1,  {0, 1}}, {2,  {0, 2}},
            {3,  {1, 0}}, {4,  {1, 1}}, {5,  {1, 2}},
            {6,  {2, 0}}, {7,  {2, 1}}, {8,  {2, 2}},
            {9,  {3, 0}}, {10, {3, 1}}, {11, {3, 2}}
        };

        const auto& mapping = (block_type == "input_blocks") ? input_map : output_map;
        auto it = mapping.find(block_idx);
        if (it == mapping.end()) return "";

        lora_block = it->second.first;
        lora_attn = it->second.second;

        if (sub_idx_str != "1") return "";

        string block_name = (block_type == "input_blocks") ? "down" : "up";
        lora_prefix = "lora_unet_" + block_name + "_blocks_" +
                      to_string(lora_block) + "_attentions_" +
                      to_string(lora_attn) + "_";
    }
    else
    {
        return "";
    }

    for (char& c : rest) if (c == '.') c = '_';
    return lora_prefix + rest;
}

struct LoRAKeySet
{
    std::string up;
    std::string down;
    std::string alpha;
};

LoRAKeySet getLoRAKeySet(const std::string& base_name)
{
    return
    {
        base_name + ".lora_up.weight",
        base_name + ".lora_down.weight",
        base_name + ".alpha"
    };
}

LoRAKeySet convertSD15KeyToLoRAKeys(const std::string& sd_key)
{
    std::string base = convertSD15KeyToLoRABase(sd_key);
    if (base.empty()) return {"", "", ""};
    return getLoRAKeySet(base);
}

#endif