#include "crypto/crypto.hpp"

namespace crypto {
  //------------------------------------------------------------------------------
  // ParseAndConstruct
  // Reads the "crypto" node from the configuration and populates a CryptoConfig struct.
  //------------------------------------------------------------------------------
  CryptoConfig* ParseAndConstruct(config::CompoundConfigNode cryptoNode) {
    CryptoConfig* cryptoCfg = new CryptoConfig;

    if (!cryptoNode.lookupValue("auth-additional-cycle-per-block", cryptoCfg->auth_additional_cycle_per_block))
      std::cerr << "Warning: 'auth-additional-cycle-per-block' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("auth-additional-energy-per-block", cryptoCfg->auth_additional_energy_per_block))
      std::cerr << "Warning: 'auth-additional-energy-per-block' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("auth-cycle-per-datapath", cryptoCfg->auth_cycle_per_datapath))
      std::cerr << "Warning: 'auth-cycle-per-datapath' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("auth-enc-parallel", cryptoCfg->auth_enc_parallel))
      std::cerr << "Warning: 'auth-enc-parallel' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("auth-energy-per-datapath", cryptoCfg->auth_energy_per_datapath))
      std::cerr << "Warning: 'auth-energy-per-datapath' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("datapath", cryptoCfg->datapath))
      std::cerr << "Warning: 'datapath' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("enc-cycle-per-datapath", cryptoCfg->enc_cycle_per_datapath))
      std::cerr << "Warning: 'enc-cycle-per-datapath' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("enc-energy-per-datapath", cryptoCfg->enc_energy_per_datapath))
      std::cerr << "Warning: 'enc-energy-per-datapath' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("family", cryptoCfg->family))
      std::cerr << "Warning: 'family' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("hash-size", cryptoCfg->hash_size))
      std::cerr << "Warning: 'hash-size' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("name", cryptoCfg->name))
      std::cerr << "Warning: 'name' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("xor-cycle", cryptoCfg->xor_cycle))
      std::cerr << "Warning: 'xor-cycle' not found. Using default value.\n";

    if (!cryptoNode.lookupValue("xor-energy-per-datapath", cryptoCfg->xor_energy_per_datapath))
      std::cerr << "Warning: 'xor-energy-per-datapath' not found. Using default value.\n";

    return cryptoCfg;
  }

} // namespace crypto