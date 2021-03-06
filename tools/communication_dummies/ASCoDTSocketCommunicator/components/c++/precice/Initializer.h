#ifndef PRECICE_INITIALIZER_H_
#define PRECICE_INITIALIZER_H_ 

//
// ASCoDT - Advanced Scientific Computing Development Toolkit
//
// This file was generated by ASCoDT's simplified SIDL compiler.
//
// Authors: Tobias Weinzierl, Atanas Atanasov   
//

#include <iostream>
#include <string>



namespace precice { 

     class Initializer;
}

class precice::Initializer {
  public:
    virtual ~Initializer(){}
     virtual void initialize(const std::string* addresses, const int addresses_len,const int* vertexes, const int vertexes_len)=0;
     virtual void initializeParallel(const std::string* addresses, const int addresses_len,const int* vertexes, const int vertexes_len)=0;
     virtual void acknowledge(const int identifier,int& tag)=0;
     virtual void acknowledgeParallel(const int identifier,int& tag)=0;


};

#endif
