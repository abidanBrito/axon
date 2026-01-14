function(add_axon_example_executable name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE axon::core)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/examples"
    )
endfunction()

add_axon_example_executable(xor_gate)
add_axon_example_executable(function_approximation)
