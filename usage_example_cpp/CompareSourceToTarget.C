// Minimal usage example for BDTReweighter.h: reweight the 0p0n/QE source
// sample event-by-event (via predict_weight_single_event, exactly as
// production analysis code calls it) and compare the reweighted source
// distribution to the target (NUISANCE flat tree) distribution, for the
// two BDT training variables (total proton KE, leading muon py) plus the
// derived psi_prime variable.
//
// Run (see SetupPybind11.C for why the two-step dance is needed --
// ACLiC needs the pybind11/Python include and link flags registered
// *before* it compiles anything that includes BDTReweighter.h):
//
//   root -l
//   root [0] .x SetupPybind11.C
//   root [1] .x CompareSourceToTarget.C+
//
// Modeled on LightSelection.cxx's reweighter_QE->GetWeight(mc, verbose)
// usage pattern inside the MC event loop, stripped of the systematics /
// error-band / HyperDimLinearizer machinery that isn't relevant here.

#include "BDTReweighter.h"

#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <iostream>

namespace {

constexpr double kProtonMassGeV = 0.93827;
constexpr int kMaxFSParticles = 100;

// Reaction-frame rotation for the muon only -- a direct port of the muon
// branch of transform_momentum_to_reaction_frame() in analysis.py: the
// muon's transverse momentum is rotated to lie entirely along -y (so its
// px component becomes 0 by construction), pz is unchanged. sum_Tp is a
// rotation-invariant scalar, so it needs no transform at all.
double MuonPyReactionFrame(double muon_px_lab, double muon_py_lab) {
    return -std::sqrt(muon_px_lab * muon_px_lab + muon_py_lab * muon_py_lab);
}

void Normalize(TH1D *h) {
    if (h->Integral() > 0) h->Scale(1.0 / h->Integral());
}

void DrawComparison(TH1D *raw, TH1D *rw, TH1D *target, const char *canvas_name, const char *png_name) {
    TCanvas *c = new TCanvas(canvas_name, canvas_name, 800, 600);
    raw->SetLineColor(kGreen + 2);
    rw->SetLineColor(kBlue + 1);
    target->SetLineColor(kRed + 1);
    raw->SetLineWidth(2);
    rw->SetLineWidth(2);
    target->SetLineWidth(2);

    double max_y = std::max({raw->GetMaximum(), rw->GetMaximum(), target->GetMaximum()});
    raw->SetMaximum(1.3 * max_y);
    raw->SetStats(0);
    raw->Draw("hist");
    rw->Draw("hist same");
    target->Draw("hist same");

    TLegend *leg = new TLegend(0.6, 0.7, 0.88, 0.88);
    leg->AddEntry(raw, "Source (raw)", "l");
    leg->AddEntry(rw, "Source (BDT reweighted)", "l");
    leg->AddEntry(target, "Target", "l");
    leg->Draw();

    c->SaveAs(png_name);
}

}  // namespace

void CompareSourceToTarget() {
    // Histograms created below must outlive the TFiles they're created
    // under (source_file->Close() runs partway through, before the
    // target-side histograms even exist) -- without this, ROOT's default
    // "histograms are owned by the current directory" behavior means
    // Close() deletes them out from under us, corrupting the heap.
    TH1::AddDirectory(kFALSE);

    // ------------------------------------------------------------------
    // Configuration -- adjust paths for your own setup.
    // ------------------------------------------------------------------
    const TString source_file_path = "/Users/lorenzo/cernbox/MINERVA_MC/source/ReweightSourceCCQELike_ABCDEFGLMNOP.root";
    const TString target_file_path = "/Users/lorenzo/cernbox/MINERVA_MC/target/neut_MINERvAflux_SF_nu_all_NUISFLAT_CCQELike.root";
    const TString model_path = "/Users/lorenzo/cernbox/MINERVA_MC/target/BDTreweight_outputs/TEST-SF/QE";
    const int reaction_code_qe = 1;      // matches processes[QE].reaction_code_rule/mode_rule: "==1"
    const Long64_t max_events = 200000;  // cap for a quick interactive check; use -1 for all events

    // api_module_dir must point at the folder containing BDTReweight_api.py.
    TString api_module_dir = gSystem->DirName(__FILE__);
    ReweighterUtils::CCQELikeBDTReweighter reweighter(model_path.Data(), api_module_dir.Data());

    // ------------------------------------------------------------------
    // Source: EventKinematics_truth. Momenta are already stored in the
    // reaction frame (that's how these reweighters were trained), so no
    // rotation is needed on this side.
    // ------------------------------------------------------------------
    TFile *source_file = TFile::Open(source_file_path);
    TTree *source_tree = (TTree *)source_file->Get("EventKinematics_truth");

    int topology, reaction_code;
    double muon_py, muon_pz, sum_Tp, init_wgt;
    source_tree->SetBranchAddress("topology", &topology);
    source_tree->SetBranchAddress("reactionCode", &reaction_code);
    source_tree->SetBranchAddress("muon_py", &muon_py);
    source_tree->SetBranchAddress("muon_pz", &muon_pz);
    source_tree->SetBranchAddress("sum_Tp", &sum_Tp);
    source_tree->SetBranchAddress("init_wgt", &init_wgt);

    TH1D *h_source_raw_ke = new TH1D("h_source_raw_ke", "total proton KE;GeV;a.u.", 25, 0.0, 0.5);
    TH1D *h_source_rw_ke = new TH1D("h_source_rw_ke", "total proton KE;GeV;a.u.", 25, 0.0, 0.5);
    TH1D *h_source_raw_py = new TH1D("h_source_raw_py", "leading muon p_{y};GeV/c;a.u.", 25, -1.0, 1.0);
    TH1D *h_source_rw_py = new TH1D("h_source_rw_py", "leading muon p_{y};GeV/c;a.u.", 25, -1.0, 1.0);
    TH1D *h_source_raw_psi = new TH1D("h_source_raw_psi", "#psi';;a.u.", 25, -5.0, 5.0);
    TH1D *h_source_rw_psi = new TH1D("h_source_rw_psi", "#psi';;a.u.", 25, -5.0, 5.0);

    Long64_t n_source = source_tree->GetEntries();
    if (max_events > 0 && max_events < n_source) n_source = max_events;
    std::cout << "Processing " << n_source << " source events..." << std::endl;

    for (Long64_t i = 0; i < n_source; ++i) {
        source_tree->GetEntry(i);
        if (topology != 0) continue;                       // 0p0n only
        if (reaction_code != reaction_code_qe) continue;    // QE process only

        // This is the exact call production C++ code makes per event:
        // GetWeight0p0n() -> Predict0p0n() -> predict_weight_single_event().
        double bdt_weight = reweighter.GetWeight0p0n(sum_Tp, muon_py, muon_pz);
        double reweighted = init_wgt * bdt_weight;
        double psi_prime = ReweighterUtils::ComputePsiPrime(sum_Tp, muon_py, muon_pz);

        h_source_raw_ke->Fill(sum_Tp, init_wgt);
        h_source_rw_ke->Fill(sum_Tp, reweighted);
        h_source_raw_py->Fill(muon_py, init_wgt);
        h_source_rw_py->Fill(muon_py, reweighted);
        h_source_raw_psi->Fill(psi_prime, init_wgt);
        h_source_rw_psi->Fill(psi_prime, reweighted);
    }
    source_file->Close();

    // ------------------------------------------------------------------
    // Target: FlatTree_VARS (NUISANCE flat tree). Final-state momenta are
    // already in the neutrino-beam frame (z = neutrino direction), so only
    // the simple in-plane rotation below is needed for the muon -- no
    // MINERvA detector beam-tilt correction, unlike raw production MC.
    // ------------------------------------------------------------------
    TFile *target_file = TFile::Open(target_file_path);
    TTree *target_tree = (TTree *)target_file->Get("FlatTree_VARS");

    int mode, nfsp;
    float px[kMaxFSParticles], py[kMaxFSParticles], pz[kMaxFSParticles], E[kMaxFSParticles];
    int pdg[kMaxFSParticles];
    float weight;
    target_tree->SetBranchAddress("Mode", &mode);
    target_tree->SetBranchAddress("nfsp", &nfsp);
    target_tree->SetBranchAddress("px", px);
    target_tree->SetBranchAddress("py", py);
    target_tree->SetBranchAddress("pz", pz);
    target_tree->SetBranchAddress("E", E);
    target_tree->SetBranchAddress("pdg", pdg);
    target_tree->SetBranchAddress("Weight", &weight);

    TH1D *h_target_ke = new TH1D("h_target_ke", "total proton KE;GeV;a.u.", 25, 0.0, 0.5);
    TH1D *h_target_py = new TH1D("h_target_py", "leading muon p_{y};GeV/c;a.u.", 25, -1.0, 1.0);
    TH1D *h_target_psi = new TH1D("h_target_psi", "#psi';;a.u.", 25, -5.0, 5.0);

    Long64_t n_target = target_tree->GetEntries();
    if (max_events > 0 && max_events < n_target) n_target = max_events;
    std::cout << "Processing " << n_target << " target events..." << std::endl;

    for (Long64_t i = 0; i < n_target; ++i) {
        target_tree->GetEntry(i);
        if (mode != reaction_code_qe) continue;  // QE process only

        int n_muons = 0, muon_idx = -1;
        double sum_tp_gev = 0.0;
        for (int j = 0; j < nfsp && j < kMaxFSParticles; ++j) {
            if (pdg[j] == 13) {
                ++n_muons;
                if (muon_idx < 0 || E[j] > E[muon_idx]) muon_idx = j;
            } else if (pdg[j] == 2212) {
                sum_tp_gev += (E[j] - kProtonMassGeV);
            }
        }
        // With the huge KE thresholds used to train these reweighters,
        // the 0p0n topology mask reduces to exactly this: one muon,
        // regardless of proton/neutron count.
        if (n_muons != 1) continue;

        double muon_py_rf = MuonPyReactionFrame(px[muon_idx], py[muon_idx]);
        double muon_pz_rf = pz[muon_idx];
        double psi_prime = ReweighterUtils::ComputePsiPrime(sum_tp_gev, muon_py_rf, muon_pz_rf);

        h_target_ke->Fill(sum_tp_gev, weight);
        h_target_py->Fill(muon_py_rf, weight);
        h_target_psi->Fill(psi_prime, weight);
    }
    target_file->Close();

    // ------------------------------------------------------------------
    // Compare shapes (each histogram normalized to unit area).
    // ------------------------------------------------------------------
    for (TH1D *h : {h_source_raw_ke, h_source_rw_ke, h_target_ke, h_source_raw_py, h_source_rw_py, h_target_py,
                    h_source_raw_psi, h_source_rw_psi, h_target_psi}) {
        Normalize(h);
    }

    DrawComparison(h_source_raw_ke, h_source_rw_ke, h_target_ke, "c_ke", "compare_total_proton_KE.png");
    DrawComparison(h_source_raw_py, h_source_rw_py, h_target_py, "c_py", "compare_leading_muon_py.png");
    DrawComparison(h_source_raw_psi, h_source_rw_psi, h_target_psi, "c_psi", "compare_psi_prime.png");

    std::cout << "Wrote compare_total_proton_KE.png, compare_leading_muon_py.png, compare_psi_prime.png" << std::endl;
}
