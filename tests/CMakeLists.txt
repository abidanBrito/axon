add_executable(ann_tests
  neuron_test.cpp
  network_test.cpp
)

target_link_libraries(ann_tests PRIVATE
  ann_core
  GTest::gtest_main
)

set_target_properties(ann_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

include(GoogleTest)
gtest_discover_tests(ann_tests)
