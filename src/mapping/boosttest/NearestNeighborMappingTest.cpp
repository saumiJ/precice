#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include "mapping/NearestNeighborMapping.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/Vertex.hpp"
#include "mesh/Data.hpp"
#include "utils/Parallel.hpp"
#include "utils/Dimensions.hpp"

// tarch::logging::Log NearestNeighborMappingTest::_log("precice::mapping::tests::NearestNeighborMappingTest");

using namespace precice::mesh;
using precice::utils::Vector2D;
using precice::utils::DynVector;

BOOST_AUTO_TEST_SUITE(NearestNeighborMapping)

BOOST_AUTO_TEST_CASE(ConsistentNonIncremental)
{
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
  DynVector& inValuesScalar = inDataScalar->values();
  DynVector& inValuesVector = inDataVector->values();
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
  const DynVector& outValuesScalar = outDataScalar->values();
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValuesScalar[0], inValuesScalar[0]);
  BOOST_CHECK_EQUAL(outValuesScalar[1], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  const DynVector& outValuesVector = outDataVector->values();
  BOOST_CHECK(tarch::la::equals(inValuesVector, outValuesVector));

  // Map data with almost coinciding vertices, has to result in equal values.
  inVertex0.setCoords(outVertex0.getCoords() + Vector2D(0.1));
  inVertex1.setCoords(outVertex1.getCoords() + Vector2D(0.1));
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValuesScalar[0], inValuesScalar[0]);
  BOOST_CHECK_EQUAL(outValuesScalar[1], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  BOOST_CHECK(tarch::la::equals(inValuesVector, outValuesVector));

  // Map data with exchanged vertices, has to result in exchanged values.
  inVertex0.setCoords(outVertex1.getCoords());
  inVertex1.setCoords(outVertex0.getCoords());
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValuesScalar[1], inValuesScalar[0]);
  BOOST_CHECK_EQUAL(outValuesScalar[0], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  DynVector expected(4);
  expected = 3.0, 4.0, 1.0, 2.0;
  BOOST_CHECK(tarch::la::equals(expected, outValuesVector));

  // Map data with coinciding output vertices, has to result in same values.
  outVertex1.setCoords(outVertex0.getCoords());
  mapping.computeMapping();
  mapping.map(inDataScalarID, outDataScalarID);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValuesScalar[1], inValuesScalar[1]);
  BOOST_CHECK_EQUAL(outValuesScalar[0], inValuesScalar[1]);
  mapping.map(inDataVectorID, outDataVectorID);
  expected = 3.0, 4.0, 3.0, 4.0;
  BOOST_CHECK(tarch::la::equals(expected, outValuesVector));
}


BOOST_AUTO_TEST_CASE(ConservativeNonIncremental)
{
  int dimensions = 2;

  // Create mesh to map from
  PtrMesh inMesh(new Mesh("InMesh", dimensions, false));
  PtrData inData = inMesh->createData("InData", 1);
  int inDataID = inData->getID();
  Vertex& inVertex0 = inMesh->createVertex(Vector2D(0.0));
  Vertex& inVertex1 = inMesh->createVertex(Vector2D(1.0));
  inMesh->allocateDataValues();
  DynVector& inValues = inData->values();
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
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), false);

  // Map data with coinciding vertices, has to result in equal values.
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  DynVector& outValues = outData->values();
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValues[0], inValues[0]);
  BOOST_CHECK_EQUAL(outValues[1], inValues[1]);
  assign(outValues) = 0.0;

  // Map data with almost coinciding vertices, has to result in equal values.
  inVertex0.setCoords(outVertex0.getCoords() + Vector2D(0.1));
  inVertex1.setCoords(outVertex1.getCoords() + Vector2D(0.1));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValues[0], inValues[0]);
  BOOST_CHECK_EQUAL(outValues[1], inValues[1]);
  assign(outValues) = 0.0;

  // Map data with exchanged vertices, has to result in exchanged values.
  inVertex0.setCoords(outVertex1.getCoords());
  inVertex1.setCoords(outVertex0.getCoords());
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValues[1], inValues[0]);
  BOOST_CHECK_EQUAL(outValues[0], inValues[1]);
  assign(outValues) = 0.0;

  // Map data with coinciding output vertices, has to result in double values.
  outVertex1.setCoords(Vector2D(-1.0));
  mapping.computeMapping();
  mapping.map(inDataID, outDataID);
  BOOST_CHECK_EQUAL(mapping.hasComputedMapping(), true);
  BOOST_CHECK_EQUAL(outValues[0], inValues[0] + inValues[1]);
  BOOST_CHECK_EQUAL(outValues[1], 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
