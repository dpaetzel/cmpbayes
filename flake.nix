{
  inputs.nixos-config.url = "github:dpaetzel/nixos-config";

  outputs = { self, nixos-config }:
    let
      nixpkgs = nixos-config.inputs.nixpkgs;
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python310;
    in rec {

      defaultPackage.${system} = python.pkgs.buildPythonPackage rec {
        pname = "cmpbayes";
        version = "1.0.0-beta";

        src = self;

        # We use pyproject.toml.
        format = "pyproject";

        buildInputs = [ pkgs.cmdstan ];

        propagatedBuildInputs = with python.pkgs; [
          arviz
          click
          matplotlib
          numpy
          pandas
          scipy
          typing-extensions

          # extra_requires
          click
          matplotlib
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
          with python.pkgs;
          [ ipython python venvShellHook ]
          ++ defaultPackage.${system}.propagatedBuildInputs;

        venvDir = "./_venv";

        postShellHook = ''
          unset SOURCE_DATE_EPOCH

          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
          }:$LD_LIBRARY_PATH";
        '';

        # Using httpstan==4.7.2 (the default as of 2022-06-10) leads to a
        # missing symbols error on NixOS. 4.7.1 works, however, so we use that.
        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install httpstan==4.7.1 pystan==3.4.0
        '';

      };
    };
}
