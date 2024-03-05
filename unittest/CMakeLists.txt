
set(${PROJECT_NAME}_SOURCES
    unittest.cpp
)
# Attempt to find Pinocchio
find_package(pinocchio REQUIRED)

# Check if Pinocchio was found
if(pinocchio_FOUND)
  message(STATUS "Pinocchio found: ${pinocchio_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "Pinocchio not found")
endif()

add_executable(${PROJECT_NAME}_unittest unittest.cpp)

# Assuming you have a target for your unit tests called grg_unittest
target_link_libraries(${PROJECT_NAME}_unittest
  PRIVATE
    ${PROJECT_NAME}  # Your main library that depends on crocoddyl, pinocchio, etc.
    pinocchio::pinocchio
    crocoddyl::crocoddyl
    example-robot-data::example-robot-data
)
