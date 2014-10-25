#pragma once
#ifndef PRECICE_NO_PETSC

#include "mapping/Mapping.hpp"
#include "RadialBasisFctMapping.hpp"
#include "boost/smart_ptr.hpp"
#include "tarch/la/DynamicMatrix.h"
#include "tarch/la/DynamicVector.h"
#include "tarch/la/LUDecomposition.h"
#include "io/TXTWriter.hpp"
#include <limits>
#include <typeinfo>

#include "petnum.hpp"
#include "petscmat.h"
#include "petscksp.h"
#include "petsclog.h"

namespace precice {
namespace mapping {

/**
 * @brief Mapping with radial basis functions using the Petsc library to solve the resulting system.
 *
 * With help of the input data points and values an interpolant is constructed.
 * The interpolant is formed by a weighted sum of conditionally positive radial
 * basis functions and a (low order) polynomial, and evaluated at the output
 * data points.
 *
 * The radial basis function type has to be given as template parameter, and has
 * to be one of the defined types in this file.
 */
template<typename RADIAL_BASIS_FUNCTION_T>
class PetRadialBasisFctMapping : public Mapping
{
public:

  /**
   * @brief Constructor.
   *
   * @param constraint [IN] Specifies mapping to be consistent or conservative.
   * @param function [IN] Radial basis function used for mapping.
   * @param solverRtol [IN] Relative tolerance for the linear solver.
   * 
   * For description on convergence testing and meaning of solverRtol see http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/KSP/KSPConvergedDefault.html#KSPConvergedDefault
   */
  PetRadialBasisFctMapping (
    Constraint              constraint,
    RADIAL_BASIS_FUNCTION_T function,
    double                  solverRtol = 1e-9);

  /// @brief Destructor, empty.
  virtual ~PetRadialBasisFctMapping()
  {
    KSPDestroy(&_solver);
  }

  /// @brief Computes the mapping coefficients from the in- and output mesh.
  virtual void computeMapping();

  /// @brief Returns true, if computeMapping() has been called.
  virtual bool hasComputedMapping();

  /// @brief Removes a computed mapping.
  virtual void clear();

  /// @brief Maps input data to output data from input mesh to output mesh.
  virtual void map ( int inputDataID, int outputDataID );

private:

  /// @brief Logging device.
  static tarch::logging::Log _log;

  bool _hasComputedMapping;

  /// @brief Radial basis function type used in interpolation.
  RADIAL_BASIS_FUNCTION_T _basisFunction;

  petsc::Matrix _matrixC;

  petsc::Matrix _matrixA;

  KSP _solver;

  double _solverRtol;
};

// --------------------------------------------------- HEADER IMPLEMENTATIONS

template<typename RADIAL_BASIS_FUNCTION_T>
tarch::logging::Log PetRadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::_log("precice::mapping::RadialBasisFctMapping");

template<typename RADIAL_BASIS_FUNCTION_T>
PetRadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::PetRadialBasisFctMapping
(
  Constraint              constraint,
  RADIAL_BASIS_FUNCTION_T function,
  double                  solverRtol)
  :
  Mapping ( constraint ),
  _hasComputedMapping ( false ),
  _basisFunction ( function ),
  _matrixC(PETSC_COMM_SELF, "C"),
  _matrixA(PETSC_COMM_SELF, "A"),
  _solverRtol(solverRtol)
{
  setInputRequirement(VERTEX);
  setOutputRequirement(VERTEX);

  KSPCreate(PETSC_COMM_SELF, &_solver);
}

template<typename RADIAL_BASIS_FUNCTION_T>
void PetRadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>::computeMapping()
{
  preciceTrace("computeMapping()");
  using namespace tarch::la;
  assertion2(input()->getDimensions() == output()->getDimensions(),
             input()->getDimensions(), output()->getDimensions());
  int dimensions = input()->getDimensions();
  mesh::PtrMesh inMesh;
  mesh::PtrMesh outMesh;
  if (getConstraint() == CONSERVATIVE){
    inMesh = output();
    outMesh = input();
  }
  else {
    inMesh = input();
    outMesh = output();
  }
  int inputSize = (int)inMesh->vertices().size();
  int outputSize = (int)outMesh->vertices().size();
  int polyparams = 1 + dimensions;
  PetscErrorCode ierr = 0;
  assertion1(inputSize >= 1 + polyparams, inputSize);
  int n = inputSize + polyparams; // Add linear polynom degrees

  _matrixC.reset(); 
  ierr = MatSetType(_matrixC.matrix, MATSBAIJ); CHKERRV(ierr); // create symmetric, block sparse matrix.
  ierr = MatSetSizes(_matrixC.matrix, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRV(ierr);
  ierr = MatSetOption(_matrixC.matrix, MAT_SYMMETRY_ETERNAL, PETSC_TRUE); CHKERRV(ierr);
  ierr = MatSetUp(_matrixC.matrix); CHKERRV(ierr);

  _matrixA.reset();
  ierr = MatSetType(_matrixA.matrix, MATAIJ); CHKERRV(ierr); // create sparse matrix.
  ierr = MatSetSizes(_matrixA.matrix, PETSC_DECIDE, PETSC_DECIDE, outputSize, n); CHKERRV(ierr);
  ierr = MatSetUp(_matrixA.matrix); CHKERRV(ierr);

  KSPReset(_solver);

  // Fill upper right part (due to symmetry) of _matrixCLU with values
  int i = 0;
  utils::DynVector distance(dimensions);
  int logCLoop = 1;
  PetscLogEventRegister("Filling Matrix C", 0, &logCLoop);
  PetscLogEventBegin(logCLoop, 0, 0, 0, 0);
  PetscInt colIdx[n];
  PetscScalar colVals[n];
  foreach (const mesh::Vertex& iVertex, inMesh->vertices()) {
    PetscInt colNum = 0;
    for (int j=iVertex.getID(); j < inputSize; j++) {
      distance = iVertex.getCoords() - inMesh->vertices()[j].getCoords();
      double coeff = _basisFunction.evaluate(norm2(distance));
      if ( not equals(coeff, 0.0)) {
        colVals[colNum] = coeff;
        colIdx[colNum] = j;
        colNum++;
      }
#     ifdef Asserts
      if (coeff == std::numeric_limits<double>::infinity()){
        preciceError("computeMapping()", "C matrix element has value inf. "
                     << "i = " << i << ", j = " << j
                     << ", coords i = " << iVertex.getCoords() << ", coords j = "
                     << inMesh->vertices()[j].getCoords() << ", dist = "
                     << distance << ", norm2 = " << norm2(distance) << ", rbf = "
                     << coeff
                     << ", rbf type = " << typeid(_basisFunction).name());
      }
#     endif
    }
    colVals[colNum] = 1;
    colIdx[colNum] = inputSize;
    colNum++;
    for (int dim=0; dim < dimensions; dim++) {
      colVals[colNum] = iVertex.getCoords()[dim];
      colIdx[colNum] = inputSize+1+dim;
      colNum++;
    }
    ierr = MatSetValues(_matrixC.matrix, 1, &i, colNum, colIdx, colVals, INSERT_VALUES); CHKERRV(ierr);
    i++;
  }
  PetscLogEventEnd(logCLoop, 0, 0, 0, 0);

  // Petsc requires that all diagonal entries are set, even if set to zero.
  _matrixC.assemble(MAT_FLUSH_ASSEMBLY);
  petsc::Vector zeros(_matrixC);
  MatDiagonalSet(_matrixC.matrix, zeros.vector, ADD_VALUES);
  ierr = MatAssemblyBegin(_matrixC.matrix, MAT_FINAL_ASSEMBLY); CHKERRV(ierr); 
  
  int logALoop = 1;
  PetscLogEventRegister("Filling Matrix A", 0, &logALoop);
  PetscLogEventBegin(logALoop, 0, 0, 0, 0);
  i = 0;
  foreach (const mesh::Vertex& iVertex, outMesh->vertices()) {
    PetscInt colNum = 0;
    int j = 0;
    foreach (const mesh::Vertex& jVertex, inMesh->vertices()) {
      distance = iVertex.getCoords() - jVertex.getCoords();
      double coeff = _basisFunction.evaluate(norm2(distance));
      if ( not equals(coeff, 0.0)) {
        colVals[colNum] = coeff;
        colIdx[colNum] = j;
        colNum++;
      }
#     ifdef Asserts
      if (coeff == std::numeric_limits<double>::infinity()){
        preciceError("computeMapping()", "A matrix element has value inf. "
                     << "i = " << i << ", j = " << j
                     << ", coords i = " << iVertex.getCoords() << ", coords j = "
                     << jVertex.getCoords() << ", dist = "
                     << distance << ", norm2 = " << norm2(distance) << ", rbf = "
                     << coeff
                     << ", rbf type = " << typeid(_basisFunction).name());
      }
#     endif
      j++;
    }
    ierr = MatSetValue(_matrixA.matrix, i, inputSize, 1.0, INSERT_VALUES); CHKERRV(ierr); 
    for (int dim=0; dim < dimensions; dim++) {
      colVals[colNum] = iVertex.getCoords()[dim];
      colIdx[colNum] = inputSize+1+dim;
      colNum++;
    }
    ierr = MatSetValues(_matrixA.matrix, 1, &i, colNum, colIdx, colVals, INSERT_VALUES); CHKERRV(ierr);
    i++;
  }
  PetscLogEventEnd(logALoop, 0, 0, 0, 0);
  ierr = MatAssemblyBegin(_matrixA.matrix, MAT_FINAL_ASSEMBLY); CHKERRV(ierr); 
  ierr = MatAssemblyEnd(_matrixC.matrix, MAT_FINAL_ASSEMBLY); CHKERRV(ierr); 
  ierr = MatAssemblyEnd(_matrixA.matrix, MAT_FINAL_ASSEMBLY); CHKERRV(ierr);

  KSPSetOperators(_solver, _matrixC.matrix, _matrixC.matrix);
  KSPSetTolerances(_solver, _solverRtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetFromOptions(_solver);

  _hasComputedMapping = true;
}

template<typename RADIAL_BASIS_FUNCTION_T>
bool PetRadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>:: hasComputedMapping()
{
  return _hasComputedMapping;
}

template<typename RADIAL_BASIS_FUNCTION_T>
void PetRadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>:: clear()
{
  preciceTrace("clear()");
  _matrixC.reset();
  _matrixA.reset();
  _hasComputedMapping = false;
}

template<typename RADIAL_BASIS_FUNCTION_T>
void PetRadialBasisFctMapping<RADIAL_BASIS_FUNCTION_T>:: map
(
  int inputDataID,
  int outputDataID )
{
  preciceTrace2("map()", inputDataID, outputDataID);
  assertion(_hasComputedMapping);
  assertion2(input()->getDimensions() == output()->getDimensions(),
             input()->getDimensions(), output()->getDimensions());
  using namespace tarch::la;
  PetscErrorCode ierr = 0;
  utils::DynVector& inValues = input()->data(inputDataID)->values();
  utils::DynVector& outValues = output()->data(outputDataID)->values();

  int valueDim = input()->data(inputDataID)->getDimensions();
  assertion2(valueDim == output()->data(outputDataID)->getDimensions(),
             valueDim, output()->data(outputDataID)->getDimensions());
  int polyparams = 1 + input()->getDimensions();


  if (getConstraint() == CONSERVATIVE) {
    preciceDebug("Map conservative");
    static int mappingIndex = 0;
    petsc::Vector Au(_matrixC, "Au");
    petsc::Vector out(_matrixC, "out");
    petsc::Vector in(_matrixA, "in");

    for (int dim=0; dim < valueDim; dim++) {
      int size = in.getSize();
      for (int i=0; i < size; i++) { // Fill input data values
        in.setValue(i, inValues[i*valueDim + dim]);
      }
      in.assemble();

      ierr = MatMultTranspose(_matrixA.matrix, in.vector, Au.vector); CHKERRV(ierr);
      ierr = KSPSolve(_solver, Au.vector, out.vector); CHKERRV(ierr);
      VecChop(out.vector, 1e-9);
      // Copy mapped data to output data values
      PetscScalar *outArray;
      ierr = VecGetArray(out.vector, &outArray);
      size = out.getSize();
      for (int i=0; i < size-polyparams; i++){
        outValues[i*valueDim + dim] = outArray[i];
      }
      VecRestoreArray(out.vector, &outArray);
    }
    mappingIndex++;
  }
  else { // Map consistent
    preciceDebug("Map consistent");
    petsc::Vector p(_matrixC, "p");
    petsc::Vector in(_matrixC, "in");
    petsc::Vector out(_matrixA, "out");
    PetscScalar *vecArray;

    // For every data dimension, perform mapping
    for (int dim=0; dim < valueDim; dim++){
      // Fill input from input data values (last polyparams entries remain zero)
      ierr = VecGetArray(in.vector, &vecArray);
      int size  = in.getSize();
      for (int i=0; i < size - polyparams; i++){
        vecArray[i] = inValues[i*valueDim + dim];
      }
      VecRestoreArray(in.vector, &vecArray);
      in.assemble();
      
      ierr = KSPSolve(_solver, in.vector, p.vector); CHKERRV(ierr);
      ierr = MatMult(_matrixA.matrix, p.vector, out.vector); CHKERRV(ierr);

      VecChop(out.vector, 1e-9);
      // Copy mapped data to output data values
      ierr = VecGetArray(out.vector, &vecArray);
      size = out.getSize();
      for (int i=0; i < size; i++) {
        outValues[i*valueDim + dim] = vecArray[i];
      }
      VecRestoreArray(out.vector, &vecArray);
    }
  }
}

}} // namespace precice, mapping

#endif