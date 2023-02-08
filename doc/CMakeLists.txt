set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)

file(GLOB AKNN_SRC
    "${PROJECT_SOURCE_DIR}/src/aknn/**.h"
)

doxygen_add_docs(aknn_docs
    "${AKNN_SRC}"
    WORKING_DIRECTORY
    "${PROJECT_SOURCE_DIR}/src/aknn"
)
