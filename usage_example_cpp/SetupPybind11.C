// Registers the include/link flags ACLiC needs to compile a translation
// unit that #includes <pybind11/embed.h> (i.e. BDTReweighter.h). Run this
// once, interpreted, at the start of a ROOT session -- BEFORE compiling
// anything that includes BDTReweighter.h -- since gSystem's include/link
// paths persist for the lifetime of the process and are consulted by every
// later ACLiC ".C+" compilation:
//
//   root -l
//   root [0] .x SetupPybind11.C
//   root [1] .x CompareSourceToTarget.C+
//
// Flags are resolved dynamically via python3-config/pybind11 rather than
// hardcoded, so this works on any machine with pybind11 installed for its
// `python3` (same interpreter the pickled reweighters were trained with).

void SetupPybind11() {
    // ACLiC only links ROOT libraries that are already loaded in the
    // current session at compile time; in batch mode (-b) the graphics
    // libraries CompareSourceToTarget.C needs (TH1D/TCanvas/TLegend)
    // aren't loaded by default, so pull them in explicitly here.
    gSystem->Load("libTree");
    gSystem->Load("libHist");
    gSystem->Load("libGraf");
    gSystem->Load("libGpad");

    TString py_includes = gSystem->GetFromPipe("python3-config --includes");
    TString py_ldflags = gSystem->GetFromPipe("python3-config --ldflags --embed");
    TString pybind_include = gSystem->GetFromPipe("python3 -c \"import pybind11; print(pybind11.get_include())\"");

    // gSystem->AddLinkedLibs() tokenizes on whitespace and can reorder
    // multi-word flags; "-framework CoreFoundation" (two tokens) trips
    // this up, so collapse it to the single-token "-Wl,-framework,..."
    // form the linker accepts equally well.
    py_ldflags.ReplaceAll("-framework CoreFoundation", "-Wl,-framework,CoreFoundation");

    if (py_includes.IsNull() || pybind_include.IsNull()) {
        std::cerr << "SetupPybind11: could not resolve python3-config/pybind11 paths. "
                  << "Is python3 (with pybind11 installed) on PATH?" << std::endl;
        return;
    }

    gSystem->AddIncludePath(Form(" -I%s %s", pybind_include.Data(), py_includes.Data()));
    gSystem->AddLinkedLibs(py_ldflags.Data());

    std::cout << "SetupPybind11: registered ACLiC include path:\n  -I" << pybind_include
              << " " << py_includes << std::endl;
    std::cout << "SetupPybind11: registered ACLiC link flags:\n  " << py_ldflags << std::endl;
}
