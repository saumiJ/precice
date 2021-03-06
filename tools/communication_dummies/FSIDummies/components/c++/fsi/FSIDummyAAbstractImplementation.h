#ifndef FSI_FSIDUMMYAABSTRACTIMPLEMENTATION_H_
#define FSI_FSIDUMMYAABSTRACTIMPLEMENTATION_H_ 

//
// ASCoDT - Advanced Scientific Computing Development Toolkit
//
// This file was generated by ASCoDT's simplified SIDL compiler.
//
// Authors: Tobias Weinzierl, Atanas Atanasov   
//
#include "fsi/FSICommNativeDispatcher.h"

#include "fsi/FSIData.h"

#include "Component.h"
namespace fsi { 

     class FSIDummyAAbstractImplementation;
}



class fsi::FSIDummyAAbstractImplementation: public Component ,public fsi::FSIData{
     protected:
       fsi::FSICommNativeDispatcher* _b;
   
     public:
       FSIDummyAAbstractImplementation();
       virtual ~FSIDummyAAbstractImplementation();
       /**
        * @see Case class 
        */
       void connectb(fsi::FSICommNativeDispatcher* port);
       void disconnectb();
 
		void dataAckParallel(int& ack);
		void transferDataParallel(const int* coordId, const int coordId_len,const double* data, const int data_len);
};     


#endif
