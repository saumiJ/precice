#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>


#include "NearestNeighborMappingTest.hpp"
#include "mapping/NearestNeighborMapping.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/Vertex.hpp"
#include "mesh/Data.hpp"
#include "utils/Parallel.hpp"
#include "utils/Dimensions.hpp"

// registerTest(precice::mapping::tests::NearestNeighborMappingTest)


// tarch::logging::Log NearestNeighborMappingTest::_log("precice::mapping::tests::NearestNeighborMappingTest");

BOOST_AUTO_TEST_SUITE(NearestNeighborMapping)

// void NearestNeighborMappingTest:: run()
// {
//   PRECICE_MASTER_ONLY {
//     testMethod(testConsistentNonIncremental);
//     testMethod(testConservativeNonIncremental);
//   }
// }

BOOST_AUTO_TEST_CASE(ConsistentNonIncremental)
{
  using namespace precice::mesh;
  using precice::utils::Vector2D;
  int dimensions = 2;

  // Create mesh to map from
  PtrMesh inMesh(new Mesh("InMesh", dimensions, false));
  PtrData inDataScalar = inMesh->createData("InDataScalar", 1);
  PtrData inDataVector = inMesh->createData("InDataVector", 2);
  int inDataScalarID = inDataScalar->getID();
  int inDataVectorID = inDataVector->getID();
  Vertex& inVertex0 = inMesh->createVertex(Vector2D(0.0));
  Vertex& inVertex1 = inMesh->createVertex(Vector2D(1.0));
  inMesh->allocateDataValues();
  precice::utils::DynVector& inValuesScalar = inDataScalar->values();
  precice::utils::DynVector& inValuesVector = inDataVector->values();
  inValuesScalar = 1.0, 2.0;
  inValuesVector = 1.0, 2.0, 3.0, 4.0;

  // Create mesh to map to
  PtrMesh outMesh(new Mesh("OutMesh", dimensions, false));
  PtrData outDataScalar = outMesh->createData("OutDataScalar", 1);
  PtrData outDataVector = outMesh->createData("OutDataVector", 2);
  int outDataScalarID = outDataScalar->getID();
  int outDataVectorID = outDataVector->getID();
  Vertex& outVertex0 = outMesh->createVertex(Vector2D(0.0));
  Vertex& outVertex1 = outMesh->createVertex(Vector2D(1.0));
  outMesh->allocateDataValues();

  // Setup mapping with mapping coordinates and geometry used
  precice::mapping::NearestNeighborMapping mapping(precice::mapping::Mapping::CONSISTENT, dimensions);
  mapping.setMeshes(inMesh, outMesh);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), false);

  // Map data with coinciding vertices, has to result in equal values.
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  const precice::utils::DynVector& outValuesScalar = outDataScalar->values();
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValuesScalar[0], inValuesScalar[0]);
  // validateNumericalEquals(outValuesScalar[1], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  const precice::utils::DynVector& outValuesVector = outDataVector->values();
  // validateWithParams2(tarch::la::equals(inValuesVector, outValuesVector), inValuesVector, outValuesVector);

  // Map data with almost coinciding vertices, has to result in equal values.
  inVertex0.setCoords(outVertex0.getCoords() + Vector2D(0.1));
  inVertex1.setCoords(outVertex1.getCoords() + Vector2D(0.1));
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValuesScalar[0], inValuesScalar[0]);
  // validateNumericalEquals(outValuesScalar[1], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  // validateWithParams2(tarch::la::equals(inValuesVector, outValuesVector), inValuesVector, outValuesVector);

  // Map data with exchanged vertices, has to result in exchanged values.
  inVertex0.setCoords(outVertex1.getCoords());
  inVertex1.setCoords(outVertex0.getCoords());
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValuesScalar[1], inValuesScalar[0]);
  // validateNumericalEquals(outValuesScalar[0], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  precice::utils::DynVector expected(4);
  expected = 3.0, 4.0, 1.0, 2.0;
  // validateWithParams2(tarch::la::equals(expected, outValuesVector), expected, outValuesVector);

  // Map data with coinciding output vertices, has to result in same values.
  outVertex1.setCoords(outVertex0.getCoords());
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValuesScalar[1], inValuesScalar[1]);
  // validateNumericalEquals(outValuesScalar[0], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  expected = 3.0, 4.0, 3.0, 4.0;
  // validateWithParams2(tarch::la::equals(expected, outValuesVector), expected, outValuesVector);
}

BOOST_AUTO_TEST_CASE(ConservativeNonIncremental)
{
  using namespace precice::mesh;
  using namespace tarch::la;
  using precice::utils::Vector2D;
  int dimensions = 2;

  // Create mesh to map from
  PtrMesh inMesh(new Mesh("InMesh", dimensions, false));
  PtrData inData = inMesh->createData("InData", 1);
  int inDataID = inData->getID();
  Vertex& inVertex0 = inMesh->createVertex(Vector2D(0.0));
  Vertex& inVertex1 = inMesh->createVertex(Vector2D(1.0));
  inMesh->allocateDataValues();
  precice::utils::DynVector& inValues = inData->values();
  inValues[0] = 1.0;
  inValues[1] = 2.0;

  // Create mesh to map to
  PtrMesh outMesh(new Mesh("OutMesh", dimensions, false));
  PtrData outData = outMesh->createData("OutData", 1);
  int outDataID = outData->getID();
  Vertex& outVertex0 = outMesh->createVertex(Vector2D(0.0));
  Vertex& outVertex1 = outMesh->createVertex(Vector2D(1.0));
  outMesh->allocateDataValues();

  // Setup mapping with mapping coordinates and geometry used
  precice::mapping::NearestNeighborMapping mapping(precice::mapping::Mapping::CONSERVATIVE, dimensions);
  mapping.setMeshes(inMesh, outMesh);
  // validateEquals(mapping.hasComputedMapping(), false);

  // Map data with coinciding vertices, has to result in equal values.
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  precice::utils::DynVector& outValues = outData->values();
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValues[0], inValues[0]);
  // validateNumericalEquals(outValues[1], inValues[1]);
  assign(outValues) = 0.0;

  // Map data with almost coinciding vertices, has to result in equal values.
  inVertex0.setCoords(outVertex0.getCoords() + Vector2D(0.1));
  inVertex1.setCoords(outVertex1.getCoords() + Vector2D(0.1));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValues[0], inValues[0]);
  // validateNumericalEquals(outValues[1], inValues[1]);
  assign(outValues) = 0.0;

  // Map data with exchanged vertices, has to result in exchanged values.
  inVertex0.setCoords(outVertex1.getCoords());
  inVertex1.setCoords(outVertex0.getCoords());
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValues[1], inValues[0]);
  // validateNumericalEquals(outValues[0], inValues[1]);
  assign(outValues) = 0.0;

  // Map data with coinciding output vertices, has to result in double values.
  outVertex1.setCoords(Vector2D(-1.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  // validateEquals(mapping.hasComputedMapping(), true);
  // validateNumericalEquals(outValues[0], inValues[0] + inValues[1]);
  // validateNumericalEquals(outValues[1], 0.0);
}


BOOST_AUTO_TEST_SUITE_END()
