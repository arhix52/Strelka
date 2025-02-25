#pragma once

class Params
{
public:
    static std::string sceneFile;
    static std::string resourceSearchPath;

    Params(){};
    ~Params(){};
};

std::string Params::sceneFile;
std::string Params::resourceSearchPath;