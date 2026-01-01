cmake_minimum_required(VERSION 3.16)

project(
  ann
  VERSION 0.0.1
  DESCRIPTION "Feed-forward artificial neural network from scratch"
  HOMEPAGE_URL "https://github.com/abidanBrito/ann"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
