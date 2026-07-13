// Minimal, standalone C++ interface to the pickled BDT reweighter models,
// embedding a Python interpreter via pybind11 and calling into
// BDTReweight_api.py -- which calls Reweighter.predict_weight_single_event()
// (../reweighter.py), the exact per-event call production analysis code
// makes for every MC event.
//
// This is a stripped-down adaptation of the production
// CCQELikeBDTReweighter_L.h header: it drops the CCQENuUtils/MINERvA
// software-stack dependency (event topology bookkeeping, beam-tilt
// correction) because the source ROOT file these reweighters were trained
// on already stores final-state momenta pre-rotated into the reaction
// frame. See CompareSourceToTarget.C for a full usage example.

#ifndef BDTREWEIGHT_USAGE_EXAMPLE_BDTREWEIGHTER_H
#define BDTREWEIGHT_USAGE_EXAMPLE_BDTREWEIGHTER_H

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ReweighterUtils {

// Physical constants for the psi_prime diagnostic variable. Must match
// train_by_reaction_config.py's MUON_MASS_GEV / NUCLEON_MASS_GEV / etc.
constexpr double kMuonMassGeV = 0.1056583745;
constexpr double kNucleonMassGeV = 0.939565;
constexpr double kSReGeV = 0.028;
constexpr double kKFermiGeV = 0.228;
constexpr double kEShiftGeV = 0.020;

// Direct C++ port of compute_psi_prime()/get_psi_prime_from_fs_kinematics()
// in train_by_reaction_config.py, specialized to muon_px == 0, which always
// holds in this codebase's reaction-frame convention (the muon is rotated
// to have zero transverse-x component by construction).
inline double ComputePsiPrime(double sum_Tp_gev, double muon_py_gev, double muon_pz_gev) {
    double muon_e = std::sqrt(muon_py_gev * muon_py_gev + muon_pz_gev * muon_pz_gev + kMuonMassGeV * kMuonMassGeV);
    double q0 = sum_Tp_gev + kSReGeV;
    double qy = -muon_py_gev;
    double q3 = muon_e - muon_pz_gev + sum_Tp_gev + kSReGeV;
    double q_mag = std::sqrt(qy * qy + q3 * q3);

    double eta_f = kKFermiGeV / kNucleonMassGeV;
    double kappa = q_mag / (2.0 * kNucleonMassGeV);
    double lambda_var = (q0 - kEShiftGeV) / (2.0 * kNucleonMassGeV);
    double tau = kappa * kappa - lambda_var * lambda_var;

    double normalizing_inner = std::sqrt(1.0 + eta_f * eta_f) - 1.0;
    if (normalizing_inner <= 0.0) return std::nan("");
    double normalizing_factor = 1.0 / std::sqrt(normalizing_inner);

    double tau_term = tau + tau * tau;
    if (tau_term < 0.0) return std::nan("");
    double denominator_sq = (1.0 + lambda_var) * tau + kappa * std::sqrt(tau_term);
    if (denominator_sq <= 0.0) return std::nan("");

    return (lambda_var - tau) / std::sqrt(denominator_sq) * normalizing_factor;
}

// Embeds a Python interpreter and loads the trained reweighter models for
// one process (e.g. QE, 2p2h, Oth) from model_path. Mirrors the production
// CCQELikeBDTReweighter_L class, but only 0p0n is wired up here since
// that's the only category trained in this codebase so far -- follow the
// same Predict<category>() pattern to add more once they exist.
class CCQELikeBDTReweighter {
public:
    // model_path:    directory holding GBReweighterModel_0p0n.pkl for one
    //                process (e.g. ".../BDTreweight_outputs/NEUT-SF/QE")
    // api_module_dir: directory containing BDTReweight_api.py (this
    //                folder, by default)
    CCQELikeBDTReweighter(std::string model_path, std::string api_module_dir)
        : f_model_path(std::move(model_path)) {
        if (!Py_IsInitialized()) py::initialize_interpreter();

        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("insert")(0, api_module_dir);
        py::module_ api = py::module_::import("BDTReweight_api");

        std::cout << "--- Initializing BDT reweighter model from " << f_model_path << std::endl;
        py::object api_cls = api.attr("BDTReweightAPI");
        f_api_instance = api_cls(f_model_path);
        f_predict_0p0n = f_api_instance.attr("predict_weight_0p0n");
    }

    // The embedded interpreter is process-wide (py::initialize_interpreter()
    // is only ever called once, guarded above); it is intentionally never
    // finalized here so that multiple CCQELikeBDTReweighter instances
    // (e.g. one per process: QE, 2p2h, Oth) can share it.
    ~CCQELikeBDTReweighter() = default;

    // Build the reaction-frame feature vector for topology 0p0n, matching
    // train_by_reaction_config.yaml's
    //   0p0n.reweight_variables: [total_proton_KE, leading_muon_py, psi_prime]
    static std::vector<double> BuildFeatures0p0n(double sum_Tp_gev, double muon_py_gev, double muon_pz_gev) {
        return {sum_Tp_gev, muon_py_gev, ComputePsiPrime(sum_Tp_gev, muon_py_gev, muon_pz_gev)};
    }

    // Per-event weight for a 0p0n event, given its reaction-frame
    // kinematics. This is the exact GetWeight()->Predict0p0n() chain that
    // production code (CCQELikeBDTReweighter_L.h) calls per MC event; the
    // only difference is that here the reaction-frame momenta are read
    // directly off the source tree instead of being computed on the fly
    // from raw detector-frame momenta via CCQENuUtils (avoid MAT libraries dependency).
    double GetWeight0p0n(double sum_Tp_gev, double muon_py_gev, double muon_pz_gev, bool verbose = false) {
        std::vector<double> features = BuildFeatures0p0n(sum_Tp_gev, muon_py_gev, muon_pz_gev);

        py::gil_scoped_acquire gil;
        double weight = f_predict_0p0n(features).cast<double>();

        if (verbose) {
            std::cout << "DEBUG: sum_Tp=" << sum_Tp_gev << ", muon_py=" << muon_py_gev
                      << ", muon_pz=" << muon_pz_gev << ", psi_prime=" << features[2]
                      << " -> weight=" << weight << std::endl;
        }
        // Guard against pathological large weights, as production does.
        if (weight > 1000.0) return 0.0;
        return weight;
    }

private:
    std::string f_model_path;
    py::object f_api_instance;
    py::object f_predict_0p0n;
};

}  // namespace ReweighterUtils

#endif  // BDTREWEIGHT_USAGE_EXAMPLE_BDTREWEIGHTER_H
