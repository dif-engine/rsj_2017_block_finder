#include "block_finder.hpp"

int main(int argc, char** argv) {
  for (int i = 0; i < argc; i++) {
    std::cout << "arg[" << i << "]: " << argv[i] << std::endl;
  }

  // 画像処理手法を選択する。
  int int_method;
  if (argc == 4) {
    int_method = std::stoi(argv[1]);
  } else {
    int_method = 2;
  }
  
  ros::init(argc, argv, "block_finder");

  // headless
  BlockFinder bf(int_method, true);

  ros::spin();
  return 0;
}
