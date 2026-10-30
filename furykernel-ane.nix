{ mkDerivation, array, base, bytestring, containers, directory
, filepath, furykernel, hedgehog, lib, tasty, tasty-hedgehog
, tasty-hunit, text, transformers
}:
mkDerivation {
  pname = "furykernel-ane";
  version = "0.2.0.0";
  src = ./.;
  libraryHaskellDepends = [
    array base bytestring containers directory filepath furykernel text
    transformers
  ];
  testHaskellDepends = [
    base bytestring containers directory filepath furykernel hedgehog
    tasty tasty-hedgehog tasty-hunit text
  ];
  homepage = "https://github.com/infurer/furykernel-ane";
  description = "The Apple Neural Engine target of FuryKernel";
  license = lib.meta.getLicenseFromSpdxId "BSD-3-Clause";
}
