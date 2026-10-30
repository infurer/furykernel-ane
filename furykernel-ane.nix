{ mkDerivation, base, bytestring, directory, filepath, hedgehog
, lib, tasty, tasty-hedgehog, tasty-hunit, transformers
}:
mkDerivation {
  pname = "furykernel-ane";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [
    base bytestring directory filepath transformers
  ];
  testHaskellDepends = [
    base bytestring directory filepath hedgehog tasty tasty-hedgehog
    tasty-hunit
  ];
  homepage = "https://github.com/infurer/furykernel-ane";
  description = "The Apple Neural Engine target of FuryKernel";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
