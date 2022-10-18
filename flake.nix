{
  inputs = {
    nixpkgs.url =
      # "github:nixos/nixpkgs/7f9b6e2babf232412682c09e57ed666d8f84ac2d";
      # Cmdstan 2.30.1
      # "github:nixos/nixpkgs/7a2d461bf2e0561bde24b8dbd6ff7676a5a68459";
      "github:NixOS/nixpkgs/0d68d7c857fe301d49cdcd56130e0beea4ecd5aa";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python39;
    in rec {

      cmdstanpy = python.pkgs.buildPythonPackage rec {
        pname = "cmdstanpy";
        version = "1.0.7";

        propagatedBuildInputs = with python.pkgs; [ numpy pandas tqdm ujson ];

        patches = [
          "${self}/0001-Remove-dynamic-cmdstan-version-selection.patch"
        ];

        postPatch = ''
          sed -i \
            "s|\(cmdstan = \)\.\.\.|\1\"${pkgs.cmdstan}/opt/cmdstan\"|" \
            cmdstanpy/utils/cmdstan.py
        '';

        doCheck = false;

        src = python.pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-AyzbqfVKup4pLl/JgDcoNKFi5te4QfO7KKt3pCNe4N8=";
        };
      };

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
