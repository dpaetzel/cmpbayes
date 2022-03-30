{
  inputs = {
    nixpkgs.url =
      "github:nixos/nixpkgs/7f9b6e2babf232412682c09e57ed666d8f84ac2d";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
      python = pkgs.python39;
    in {
      defaultPackage.x86_64-linux = python.pkgs.buildPythonPackage rec {
        pname = "cmpbayes";
        version = "0.0.1-beta";

        src = self;

        # We use pyproject.toml.
        format = "pyproject";

        # TODO In order to provide a proper default flake here we need to
        # package pystan/httpstan properly. For now, we assume that pystan is
        # already there.
        postPatch = ''
          sed -i "s/^.*pystan.*$//" setup.cfg
        '';

        propagatedBuildInputs = with python.pkgs; [
            arviz
            matplotlib
            numpy
            pandas
        ];

        meta = with pkgs.lib; {
          description =
            "Small Python library for Bayesian data analysis for algorithms results";
          license = licenses.gpl3;
        };
      };

      devShell.${system} = pkgs.mkShell {

        # We use ugly venvShellHook here because packaging pystan/httpstan is
        # not entirely straightforward.
        buildInputs = with pkgs;
          with python.pkgs; [
            python
            venvShellHook
            numpy
            pandas
          ];

        venvDir = "./_venv";

        postShellHook = ''
          unset SOURCE_DATE_EPOCH

          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
          }:$LD_LIBRARY_PATH";
        '';

        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install pystan==3.4.0
        '';

      };
    };
}
