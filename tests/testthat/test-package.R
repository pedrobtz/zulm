test_that("the package declares its family imports", {
  imports <- getNamespaceImports("zulm")
  expect_true(all(c("zujson", "zurand") %in% names(imports)))
})
