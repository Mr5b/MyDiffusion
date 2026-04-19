#ifndef PROMPT_WEIGHT_PARSER_HPP
#define PROMPT_WEIGHT_PARSER_HPP

#include <vector>
#include <string>
#include <stack>
#include <regex>
#include <cctype>
#include <codecvt>
#include <locale>
#include <algorithm>
#include "tokenizer.hpp"

class ExtendedCLIPTokenizer : public MNN::DIFFUSION::CLIPTokenizer
{
public:

    std::pair<std::vector<int>, std::vector<float>> encode_weighted(const std::string& prompt, int maxlen)
    {
        auto [clean_prompt, subword_weights] = parse_weighted_prompt(prompt);

        
        std::vector<int> ids = this->CLIPTokenizer::encode(clean_prompt, maxlen);

        
        std::vector<float> weights(2 * maxlen, 1.0f);


        int start_pos = maxlen;
        int sos_pos = start_pos;
        int eos_pos = start_pos + 1 + subword_weights.size();
        
        int weight_idx = 0;
        for (int i = start_pos + 1; i < 2 * maxlen && weight_idx < subword_weights.size(); i++)
        {
            if (ids[i] == 0) break;
            if (ids[i] == mEndIdx) break;
            weights[i] = subword_weights[weight_idx++];
        }

        return {ids, weights};
    }

//private:
    struct WeightedToken
    {
        std::string token;
        float weight;
    };
    
    std::pair<std::string, std::vector<float>> parse_weighted_prompt(const std::string& prompt)
    {
        std::vector<WeightedToken> weighted_tokens = parse_weighted_tokens(prompt);
        
        std::string clean;
        for (size_t i = 0; i < weighted_tokens.size(); i++)
        {
            if (i != 0) clean += ' ';
            clean += weighted_tokens[i].token;
        }

        std::vector<std::wstring> subwords = tokenize_to_subwords(clean);


        std::vector<float> subword_weights;
        size_t token_idx = 0;
        std::string current_token_str = weighted_tokens[token_idx].token;
        std::string accumulated;
        float current_weight = weighted_tokens[token_idx].weight;

        for (const auto& sw : subwords)
        {
            std::string sw_utf8 = wstring_to_utf8(sw);
            if (sw_utf8.size() >= 4 && sw_utf8.substr(sw_utf8.size()-4) == "</w>")
            {
                sw_utf8 = sw_utf8.substr(0, sw_utf8.size()-4);
            }
            accumulated += sw_utf8;
            
            subword_weights.push_back(current_weight);
    
            if (accumulated == current_token_str)
            {
                token_idx++;
                if (token_idx < weighted_tokens.size())
                {
                    current_token_str = weighted_tokens[token_idx].token;
                    current_weight = weighted_tokens[token_idx].weight;
                    accumulated.clear();
                }
            }
        }

        while (subword_weights.size() < subwords.size())
        {
            subword_weights.push_back(1.0f);
        }

        return {clean, subword_weights};
    }

    
    std::vector<struct WeightedToken> parse_weighted_tokens(const std::string& prompt)
    {
        std::vector<WeightedToken> result;
        std::stack<float> weight_stack;
        weight_stack.push(1.0f);

        size_t i = 0;
        const size_t len = prompt.size();

        auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)); };
        auto is_alnum = [](char c) { return std::isalnum(static_cast<unsigned char>(c)); };

        while (i < len)
        {
            char c = prompt[i];
            if (c == '(')
            {
                size_t end = prompt.find(')', i);
                if (end != std::string::npos)
                {
                    std::string content = prompt.substr(i+1, end - i - 1);
                    size_t colon = content.find(':');
                    if (colon != std::string::npos)
                    {
                        std::string keyword = content.substr(0, colon);
                        float w = std::stof(content.substr(colon+1));
                        float current_weight = weight_stack.top() * w;
                        if (!keyword.empty())
                        {
                            result.push_back({keyword, current_weight});
                        }
                        i = end + 1;
                        continue;
                    }
                }
                
                weight_stack.push(weight_stack.top() * 1.1f);
                i++;
            }
            else if (c == '[')
            {
                weight_stack.push(weight_stack.top() * 0.9091f);
                i++;
            }
            else if (c == ')' || c == ']')
            {
                if (weight_stack.size() > 1) weight_stack.pop();
                i++;
            }
            else if (is_space(c))
            {
                i++;
                continue;
            }
            else
            {
                std::string token;
                while
                (
                    i < len &&
                    !is_space(prompt[i]) &&
                    prompt[i] != '(' &&
                    prompt[i] != ')' &&
                    prompt[i] != '[' &&
                    prompt[i] != ']'
                )
                {
                    token += prompt[i];
                    i++;
                }
                if (!token.empty())
                {
                    result.push_back({token, weight_stack.top()});
                }
            }
        }
        return result;
    }
    
    std::string wstring_to_utf8(const std::wstring& wstr)
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        return conv.to_bytes(wstr);
    }


    std::vector<std::wstring> tokenize_to_subwords(const std::string& text)
    {
        std::regex re(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[[:alpha:]]+|[[:digit:]]|[^[:space:][:alpha:][:digit:]]+)",
                      std::regex::icase);
        std::string input = text;
        std::vector<std::wstring> result;
        std::string token;
        std::smatch match;
        while (std::regex_search(input, match, re))
        {
            token = match.str(0);
            input = match.suffix().str();

            std::wstring wtoken;
            for (char c : token)
            {
                wtoken.push_back(b2u_.at(uint8_t(c)));
            }
            std::vector<std::wstring> bpe_tokens;
            bpe(wtoken, bpe_ranks_, &bpe_tokens);
            result.insert(result.end(), bpe_tokens.begin(), bpe_tokens.end());
        }
        return result;
    }
};

#endif // PROMPT_WEIGHT_PARSER_HPP