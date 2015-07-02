// Copyright (C) 2011 Technische Universitaet Muenchen
// This file is part of the preCICE project. For conditions of distribution and
// use, please see the license notice at http://www5.in.tum.de/wiki/index.php/PreCICE_License
#include "IQNILSPostProcessing.hpp"
#include "cplscheme/CouplingData.hpp"
#include "utils/Globals.hpp"
#include "tarch/la/GramSchmidt.h"
#include "tarch/la/MatrixVectorOperations.h"
#include "tarch/la/TransposedMatrix.h"
#include "mesh/Mesh.hpp"
#include "mesh/Vertex.hpp"
#include "utils/Dimensions.hpp"
#include "tarch/la/Scalar.h"
#include "io/TXTWriter.hpp"
#include "io/TXTReader.hpp"
#include "utils/MasterSlave.hpp"
#include "QRFactorization.hpp"
#include "Eigen/Dense"

#include "tarch/tests/TestMacros.h"

#include <time.h>

#include <unistd.h>

//#include "utils/NumericalCompare.hpp"

namespace precice {
namespace cplscheme {
namespace impl {

// tarch::logging::Log IQNILSPostProcessing::
//       _log("precice::cplscheme::impl::IQNILSPostProcessing");

IQNILSPostProcessing:: IQNILSPostProcessing
(
  double initialRelaxation,
  int    maxIterationsUsed,
  int    timestepsReused,
  std::string filter,
  double singularityLimit,
  std::vector<int> dataIDs,
  std::map<int,double> scalings)
:
  BaseQNPostProcessing(initialRelaxation, maxIterationsUsed, timestepsReused,
		       filter, singularityLimit, dataIDs, scalings),
  _secondaryOldXTildes(),
  _secondaryMatricesW()
{
}

void IQNILSPostProcessing:: initialize
(
  DataMap& cplData )
{
  // do common QN post processing initialization
  BaseQNPostProcessing::initialize(cplData);

  double init = 0.0;
  // Fetch secondary data IDs, to be relaxed with same coefficients from IQN-ILS
  foreach (DataMap::value_type& pair, cplData){
    if (not utils::contained(pair.first, _dataIDs)){
      int secondaryEntries = pair.second->values->size();
      _secondaryOldXTildes[pair.first].append(DataValues(secondaryEntries, init));
    }
  }
  
}


void IQNILSPostProcessing::updateDifferenceMatrices
(
  DataMap& cplData)
{
  // Compute residuals of secondary data
  foreach (int id, _secondaryDataIDs){
    DataValues& secResiduals = _secondaryResiduals[id];
    PtrCouplingData data = cplData[id];
    assertion2(secResiduals.size() == data->values->size(),
               secResiduals.size(), data->values->size());
    secResiduals = *(data->values);
    secResiduals -= data->oldValues.column(0);
  }

  /*
   * ATTETION: changed the condition from _firstIteration && _firstTimeStep
   * to the following: 
   * underrelaxation has to be done, if the scheme has converged without even
   * entering post processing. In this case the V, W matrices would still be empty.
   * This case happended in the open foam example beamInCrossFlow.
   */ 
  if(_firstIteration && (_firstTimeStep ||  (_matrixCols.size() < 2))){
//     // Store x_tildes for secondary data
//     foreach (int id, _secondaryDataIDs){
//       assertion2(_secondaryOldXTildes[id].size() == cplData[id]->values->size(),
//                  _secondaryOldXTildes[id].size(), cplData[id]->values->size());
//       _secondaryOldXTildes[id] = *(cplData[id]->values);
//     }
// 
//     // Perform underrelaxation with initial relaxation factor for secondary data
//     foreach (int id, _secondaryDataIDs){
//       PtrCouplingData data = cplData[id];
//       DataValues& values = *(data->values);
//       values *= _initialRelaxation;                   // new * omg
//       DataValues& secResiduals = _secondaryResiduals[id];
//       secResiduals = data->oldValues.column(0);    // old
//       secResiduals *= 1.0 - _initialRelaxation;       // (1-omg) * old
//       values += secResiduals;                      // (1-omg) * old + new * omg
//     }
  }
  else {
    if (not _firstIteration){
      bool columnLimitReached = _matrixV.cols() == _maxIterationsUsed;
      bool overdetermined = _matrixV.cols() <= _matrixV.rows();
      if (not columnLimitReached && overdetermined){
        
	// Append column for secondary W matrices
        foreach (int id, _secondaryDataIDs){
          _secondaryMatricesW[id].appendFront(_secondaryResiduals[id]);
        }
      }
      else {
        // Shift column for secondary W matrices
        foreach (int id, _secondaryDataIDs){
          _secondaryMatricesW[id].shiftSetFirst(_secondaryResiduals[id]);
        }
      }

      // Compute delta_x_tilde for secondary data
      foreach (int id, _secondaryDataIDs){
        DataMatrix& secW = _secondaryMatricesW[id];
        assertion2(secW.column(0).size() == cplData[id]->values->size(),
                   secW.column(0).size(), cplData[id]->values->size());
        secW.column(0) = *(cplData[id]->values);
        secW.column(0) -= _secondaryOldXTildes[id];
      }
    }

    // Store x_tildes for secondary data
    foreach (int id, _secondaryDataIDs){ 
      assertion2(_secondaryOldXTildes[id].size() == cplData[id]->values->size(),
                 _secondaryOldXTildes[id].size(), cplData[id]->values->size());
      _secondaryOldXTildes[id] = *(cplData[id]->values);
    }
  }
  
  
  // call the base method for common update of V, W matrices
  BaseQNPostProcessing::updateDifferenceMatrices(cplData);
}


void IQNILSPostProcessing::computeUnderrelaxationSecondaryData
(
  DataMap& cplData)
{
    //Store x_tildes for secondary data
    foreach (int id, _secondaryDataIDs){
      assertion2(_secondaryOldXTildes[id].size() == cplData[id]->values->size(),
                 _secondaryOldXTildes[id].size(), cplData[id]->values->size());
      _secondaryOldXTildes[id] = *(cplData[id]->values);
    }

    // Perform underrelaxation with initial relaxation factor for secondary data
    foreach (int id, _secondaryDataIDs){
      PtrCouplingData data = cplData[id];
      DataValues& values = *(data->values);
      values *= _initialRelaxation;                   // new * omg
      DataValues& secResiduals = _secondaryResiduals[id];
      secResiduals = data->oldValues.column(0);    // old
      secResiduals *= 1.0 - _initialRelaxation;       // (1-omg) * old
      values += secResiduals;                      // (1-omg) * old + new * omg
    }
}


void IQNILSPostProcessing::computeQNUpdate_PODFilter
(PostProcessing::DataMap& cplData, DataValues& xUpdate)
{
	preciceTrace("computeQNUpdate_PODFilter()");

	// copy matrix V to Eigen Matrix data type
	Eigen::MatrixXd _V(_matrixV.rows(), _matrixV.cols());
	for (int i = 0; i < _V.rows(); i++)
		for (int j = 0; j < _V.cols(); j++) {
			_V(i, j) = _matrixV(i, j);
		}
	// copy matrix W to Eigen Matrix data type
	Eigen::MatrixXd _W(_matrixW.rows(), _matrixW.cols());
	for (int i = 0; i < _W.rows(); i++)
		for (int j = 0; j < _W.cols(); j++) {
			_W(i, j) = _matrixW(i, j);
		}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(_V, Eigen::ComputeFullV);
	Eigen::VectorXd sigma = svd.singularValues();
	Eigen::MatrixXd phi = svd.matrixV();

	//assertion2(sigma.size() == _matrixV.cols(), sigma.size(), _matrixV.cols);
	assertion2(phi.rows() == _V.cols(), phi.rows(), _V.cols());
	assertion2(phi.rows() == _W.cols(), phi.rows(), _W.cols());

	/*
	{
	    int i = 0;
	    char hostname[256];
	    gethostname(hostname, sizeof(hostname));
	    printf("PID %d on %s ready for attach\n", getpid(), hostname);
	    fflush(stdout);
	    while (0 == i)
	        sleep(5);
	}
	*/

	int k = _V.cols();
	double lambda_1 = sigma(0)*sigma(0)/phi.rows();
	for(int i = 1; i < sigma.size(); i++)
	{
		double lambda_i = sigma(i)*sigma(i)/phi.rows();
		if(lambda_i/lambda_1 <= _singularityLimit)
		{
			k = i;

			// print
			preciceDebug("   (POD-Filter) t="<<tSteps<<", k="<<its
					    <<" | truncating matrices VX, WX after the first " << k
					    <<"columns. Discarded columns: "<< _V.cols()-k);
			_infostream << "   (POD-Filter) t="<<tSteps<<", k="<<its
						<<" | truncating matrices VX, WX after the first " << k
						<<"columns. Discarded columns: "<< _V.cols()-k<< std::flush<<std::endl;
			// debugging information, can be removed
			deletedColumns += _V.rows()-k;
			break;
		}
	}

	// compute V_til = V*phi and W_til = W*phi
    _V = _V*phi;
    _W = _W*phi;

    // truncate
    _V.conservativeResize(_V.rows(), k);
    _W.conservativeResize(_W.rows(), k);


    DataMatrix Vcopy(_V.rows(), _V.cols(), 0.0);
    for (int i = 0; i < _V.rows(); i++)
    		for (int j = 0; j < _V.cols(); j++) {
    			Vcopy(i, j) = _V(i, j);
    		}
    DataMatrix Wcopy(_W.rows(), _W.cols(), 0.0);
	for (int i = 0; i < _W.rows(); i++)
			for (int j = 0; j < _W.cols(); j++) {
				Wcopy(i, j) = _W(i, j);
			}
	DataMatrix Q(Vcopy.rows(), Vcopy.cols(), 0.0);
	DataMatrix R(Vcopy.cols(), Vcopy.cols(), 0.0);
	tarch::la::modifiedGramSchmidt(Vcopy, Q, R);

	DataValues c;
	DataValues b(Q.cols(), 0.0);
	tarch::la::multiply(tarch::la::transpose(Q), _residuals, b); // = Qr
    b *= -1.0; // = -Qr
    assertion1(c.size() == 0, c.size());
    c.append(b.size(), 0.0);

    tarch::la::backSubstitution(R, b, c);
    tarch::la::multiply(Wcopy, c, xUpdate);

	preciceDebug("c = " << c);

	/**
	 * ATTENTION:
	 * TODO: the truncated matrices are not copied back to the tarch matrices _matrixV, _matrixW
	 *       also for the secondaryMatrices here is some work to do. The following is incorrect,
	 *       and needs some re-thinking.
	 */

 /*
	// Perform QN relaxation for secondary data
	foreach (int id, _secondaryDataIDs){
		PtrCouplingData data = cplData[id];
		DataValues& values = *(data->values);
		assertion2(_secondaryMatricesW[id].cols() == c.size(),
				_secondaryMatricesW[id].cols(), c.size());
		tarch::la::multiply(_secondaryMatricesW[id], c, values);
		assertion2(values.size() == data->oldValues.column(0).size(),
				values.size(), data->oldValues.column(0).size());
		values += data->oldValues.column(0);
		assertion2(values.size() == _secondaryResiduals[id].size(),
				values.size(), _secondaryResiduals[id].size());
		values += _secondaryResiduals[id];
	}
*/

}

void IQNILSPostProcessing::computeQNUpdate_QRFilter1
(PostProcessing::DataMap& cplData, DataValues& xUpdate)
{
	preciceTrace("computeQNUpdate_QRFilter1()");
	using namespace tarch::la;

	// Calculate QR decomposition of matrix V and solve Rc = -Qr
	//DataValues c;
	DataValues __c;
	bool linearDependence = true;
	while (linearDependence) {
		preciceDebug("   Compute Newton factors");
		linearDependence = false;

		Matrix __R(_matrixV.cols(), _matrixV.cols(), 0.0);
		auto r = _qrV.matrixR();
		for (int i = 0; i < r.rows(); i++)
			for (int j = 0; j < r.cols(); j++) {
				__R(i, j) = r(i, j);
			}

		if (_matrixV.cols() > 1) {
			for (int i = 0; i < _matrixV.cols(); i++) {
				if (std::fabs(__R(i, i)) < _singularityLimit) {

					preciceDebug("   (QR1-Filter) t="<<tSteps<<", k="<<its
							    <<" | deleting column " << i );
					_infostream <<"   (QR1-Filter) t="<<tSteps<<", k="<<its
						    	<<" | deleting column " << i << std::flush<<std::endl;

					linearDependence = true;
					removeMatrixColumn(i);
				}
			}
		}
		if (not linearDependence) {

			preciceDebug("   Apply Newton factors");

			// --------- QN factors with modifiedGramSchmidt ---
			/*
			 DataMatrix Vcopy(_matrixV);
			 DataMatrix Q(Vcopy.rows(), Vcopy.cols(), 0.0);
			 DataMatrix R(Vcopy.cols(), Vcopy.cols(), 0.0);
			 modifiedGramSchmidt(Vcopy, Q, R);

			 DataValues b(Q.cols(), 0.0);
			 multiply(transpose(Q), _residuals, b); // = Qr
			 b *= -1.0; // = -Qr
			 assertion1(c.size() == 0, c.size());
			 c.append(b.size(), 0.0);

			 backSubstitution(R, b, c);

			 DataValues update(_residuals.size(), 0.0);
			 multiply(_matrixW, c, update); // = Wc
			 */

			// ---------- QN factors with updatedQR -----------
			Matrix __Qt(_matrixV.cols(), _matrixV.rows(), 0.0);

			auto q = _qrV.matrixQ();
			for (int i = 0; i < q.rows(); i++)
				for (int j = 0; j < q.cols(); j++) {
					__Qt(j, i) = q(i, j);
				}
			auto r = _qrV.matrixR();
			for (int i = 0; i < r.rows(); i++)
				for (int j = 0; j < r.cols(); j++) {
					__R(i, j) = r(i, j);
				}

			DataValues __b(__Qt.rows(), 0.0);
			multiply(__Qt, _residuals, __b);
			__b *= -1.0; // = -Qr
			assertion1(__c.size() == 0, __c.size());
			__c.append(__b.size(), 0.0);
			backSubstitution(__R, __b, __c);

			multiply(_matrixW, __c, xUpdate);

			preciceDebug("c = " << __c);

			// Perform QN relaxation for secondary data
			foreach (int id, _secondaryDataIDs){
				PtrCouplingData data = cplData[id];
				DataValues& values = *(data->values);
				assertion2(_secondaryMatricesW[id].cols() == __c.size(),
						_secondaryMatricesW[id].cols(), __c.size());
				tarch::la::multiply(_secondaryMatricesW[id], __c, values);
				assertion2(values.size() == data->oldValues.column(0).size(),
						values.size(), data->oldValues.column(0).size());
				values += data->oldValues.column(0);
				assertion2(values.size() == _secondaryResiduals[id].size(),
						values.size(), _secondaryResiduals[id].size());
				values += _secondaryResiduals[id];
			}
		}
	}
}

void IQNILSPostProcessing::computeQNUpdate_QRFilter2
(PostProcessing::DataMap& cplData, DataValues& xUpdate)
{
	preciceTrace("computeQNUpdate_QRFilter2()");
	bool termination = false;
	while(!termination)
	{
		// copy matrix V to Eigen Matrix data type
		Eigen::MatrixXd _V(_matrixV.rows(), _matrixV.cols());
		for (int i = 0; i < _V.rows(); i++)
			for (int j = 0; j < _V.cols(); j++) {
				_V(i, j) = _matrixV(i, j);
			}
		// copy matrix W to Eigen Matrix data type
		Eigen::MatrixXd _W(_matrixW.rows(), _matrixW.cols());
		for (int i = 0; i < _W.rows(); i++)
			for (int j = 0; j < _W.cols(); j++) {
				_W(i, j) = _matrixW(i, j);
			}

		termination = true;

		// compute QR-decomposition of V
		Eigen::MatrixXd _Q(_matrixV.rows(), _matrixV.cols());
		Eigen::MatrixXd _R(_matrixV.cols(), _matrixV.cols());

		Eigen::VectorXd v0 = _V.col(0);
		_R(0,0) = v0.norm();
		_Q.col(0) = v0/_R(0,0);
		for( int j=1; j<_V.cols(); j++)
		{
			Eigen::VectorXd v = _V.col(j);
			double rho0 = v.norm();

			for(int i=0; i < j; i++)
			{
				Eigen::VectorXd Qci = _Q.col(i);
				_R(i,j) = Qci.dot(v);
				v = v -_R(i,j)*Qci;
			}

			// QR-filter test (if information that comes with vector v
			// is little, i.e., |v_orth| small, discard vector v.)
			double rho1 = v.norm();
			//_infostream <<"  rho1:"<<rho1<<", eps*rho0:"<<rho0*_singularityLimit<<std::endl;
			if(rho1 < _singularityLimit * rho0)
			{
				termination = false;
				removeMatrixColumn(j);

				preciceDebug("   (QR2-Filter) t="<<tSteps<<", k="<<its<<" | deleting column " << j );
				_infostream <<"   (QR2-Filter) t="<<tSteps<<", k="<<its
							<<" | deleting column " << j << std::flush<<std::endl;


				break;
			}

			// normalize
			_R(j,j) = rho1;
			_Q.col(j) = v/rho1;
		}

		if(termination)
		{
			// copy back, Q and R
			DataMatrix Q(_Q.rows(), _Q.cols(), 0.0);
			for (int i = 0; i < _Q.rows(); i++)
						for (int j = 0; j < _Q.cols(); j++) {
							Q(i, j) = _Q(i, j);
						}
			DataMatrix R(_R.rows(), _R.cols(), 0.0);
			for (int j = 0; j < _R.cols(); j++)
						for (int i = 0; i <= j; i++) {
							R(i, j) = _R(i, j);
						}

			// compute update
			DataValues c;
			DataValues b(Q.cols(), 0.0);
			tarch::la::multiply(tarch::la::transpose(Q), _residuals, b); // = Qr
			b *= -1.0; // = -Qr
			assertion1(c.size() == 0, c.size());
			c.append(b.size(), 0.0);

			tarch::la::backSubstitution(R, b, c);
			tarch::la::multiply(_matrixW, c, xUpdate);

			preciceDebug("c = " << c);

			// Perform QN relaxation for secondary data
			foreach (int id, _secondaryDataIDs){
				PtrCouplingData data = cplData[id];
				DataValues& values = *(data->values);
				assertion2(_secondaryMatricesW[id].cols() == c.size(),
						_secondaryMatricesW[id].cols(), c.size());
				tarch::la::multiply(_secondaryMatricesW[id], c, values);
				assertion2(values.size() == data->oldValues.column(0).size(),
						values.size(), data->oldValues.column(0).size());
				values += data->oldValues.column(0);
				assertion2(values.size() == _secondaryResiduals[id].size(),
						values.size(), _secondaryResiduals[id].size());
				values += _secondaryResiduals[id];
			}
		}
	}
}


void IQNILSPostProcessing::computeQNUpdate
(PostProcessing::DataMap& cplData, DataValues& xUpdate)
{
	preciceTrace("computeQNUpdate()");


    if(_filter == QR1_FILTER)
    {
    	computeQNUpdate_QRFilter1(cplData, xUpdate);
    }else if(_filter == QR2_FILTER)
    {
    	computeQNUpdate_QRFilter2(cplData, xUpdate);
    }else if(_filter == POD_FILTER)
    {
    	computeQNUpdate_PODFilter(cplData, xUpdate);
    }else
    {
    	preciceError("computeQNUpdate()", "invalid or no filter specified for least-squares system of QN-post-processing.");
    }
}


void IQNILSPostProcessing:: specializedIterationsConverged
(
   DataMap & cplData)
{
  
  if (_matrixCols.front() == 0){ // Did only one iteration
    _matrixCols.pop_front(); 
  }
  
  if (_timestepsReused == 0){
    foreach (int id, _secondaryDataIDs){
      _secondaryMatricesW[id].clear();
    }
  }
  else if ((int)_matrixCols.size() > _timestepsReused){
    int toRemove = _matrixCols.back();
    foreach (int id, _secondaryDataIDs){
      DataMatrix& secW = _secondaryMatricesW[id];
      assertion3(secW.cols() > toRemove, secW, toRemove, id);
      for (int i=0; i < toRemove; i++){
        secW.remove(secW.cols() - 1);
      }
    }
  }
  
}


void IQNILSPostProcessing:: removeMatrixColumn
(
  int columnIndex)
{
  assertion(_matrixV.cols() > 1);
  // remove column from secondary Data Matrix W
 foreach (int id, _secondaryDataIDs){
    _secondaryMatricesW[id].remove(columnIndex);
  }
  
  BaseQNPostProcessing::removeMatrixColumn(columnIndex);
}

}}} // namespace precice, cplscheme, impl
