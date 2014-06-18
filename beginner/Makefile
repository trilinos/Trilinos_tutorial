subsystem:
	$(MAKE) -C Anasazi_Block_Davidson
	$(MAKE) -C Anasazi_Block_KrylovSchur 
	$(MAKE) -C Anasazi_LOBPCG 
	$(MAKE) -C Epetra_Power_Method 
	$(MAKE) -C Epetra_Simple_Vector 
	$(MAKE) -C Epetra_Lesson01-Init
	$(MAKE) -C Epetra_Lesson02-Map-Vector
	$(MAKE) -C Epetra_Lesson03-Power-Method
#	$(MAKE) -C Epetra_Lesson04-Sparse-Matrix-Fill
	$(MAKE) -C Epetra_Lesson05-Redistribution
	$(MAKE) -C Galeri_Linear_System 
	$(MAKE) -C Ifpack_Preconditioner_Factory 
	$(MAKE) -C Linear_Solver_Belos 
	$(MAKE) -C Linear_Solver_Ifpack 
	$(MAKE) -C Linear_Solver_ml 
	$(MAKE) -C Linear_Solver_mlMultiGrid 
	$(MAKE) -C NOX_Newton1 
	$(MAKE) -C NOX_Newton2 
	$(MAKE) -C Teuchos_BLAS
	$(MAKE) -C Teuchos_CLP 
	$(MAKE) -C Teuchos_LAPACK 
	$(MAKE) -C Teuchos_PL  
	$(MAKE) -C Teuchos_RCP 
	$(MAKE) -C Teuchos_SDM 
	$(MAKE) -C Teuchos_Time 
	$(MAKE) -C Tpetra_Init 
	$(MAKE) -C Tpetra_Vector 
	$(MAKE) -C Tpetra_Lesson01-Init
	$(MAKE) -C Tpetra_Lesson02-Map-Vector
	$(MAKE) -C Tpetra_Lesson03-Power-Method
#	$(MAKE) -C Tpetra_Lesson04-Sparse-Matrix-Fill
	$(MAKE) -C Tpetra_Lesson05-Redistribution

SUBDIRS = Anasazi_Block_Davidson Anasazi_Block_KrylovSchur Anasazi_LOBPCG Epetra_Power_Method Epetra_Simple_Vector Epetra_Lesson01-Init Epetra_Lesson02-Map-Vector Epetra_Lesson03-Power-Method Epetra_Lesson05-Redistribution Galeri_Linear_System Ifpack_Preconditioner_Factory Linear_Solver_Belos Linear_Solver_Ifpack Linear_Solver_ml Linear_Solver_mlMultiGrid NOX_Newton1 NOX_Newton2 Teuchos_BLAS Teuchos_CLP Teuchos_LAPACK Teuchos_PL Teuchos_RCP Teuchos_SDM Teuchos_Time Tpetra_Init Tpetra_Vector Tpetra_Lesson01-Init Tpetra_Lesson02-Map-Vector Tpetra_Lesson03-Power-Method Tpetra_Lesson05-Redistribution

.PHONY: clean $(SUBDIRS)

clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) clean -C $@

