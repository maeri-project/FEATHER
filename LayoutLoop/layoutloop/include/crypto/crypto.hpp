#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdint>

// Include the compound-config API using the new header path.
#include "compound-config/compound-config.hpp"  // Definitions in namespace config


namespace crypto {

  //------------------------------------------------------------------------------
  // CryptoConfig structure: holds parameters from the crypto YAML node.
  //------------------------------------------------------------------------------
  struct CryptoConfig {
    int auth_additional_cycle_per_block = 0;
    int auth_additional_energy_per_block = 0;
    int auth_cycle_per_datapath = 0;
    bool auth_enc_parallel = false;
    int auth_energy_per_datapath = 0;
    int datapath = 0;
    int enc_cycle_per_datapath = 0;
    int enc_energy_per_datapath = 0;
    std::string family;
    int hash_size = 0;
    std::string name;
    int xor_cycle = 0;
    int xor_energy_per_datapath = 0;
    bool crypto_initialized_ = false;
  };
  
  //------------------------------------------------------------------------------
  // ParseAndConstruct
  // Reads the "crypto" node from the configuration and populates a CryptoConfig struct.
  //------------------------------------------------------------------------------
  CryptoConfig* ParseAndConstruct(config::CompoundConfigNode rootNode);

} // namespace crypto