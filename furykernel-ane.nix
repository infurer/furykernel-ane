{ mkDerivation, base, lib }:
mkDerivation {
  pname = "furykernel-ane";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [ base ];
  homepage = "https://github.com/infurer/furykernel-ane";
  description = "The Apple Neural Engine target of FuryKernel";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
