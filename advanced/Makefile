subsystem:
	$(MAKE) -C Epetra_Basic_Perf
	$(MAKE) -C Stratimikos_Solver_Driver
	$(MAKE) -C Stratimikos_Preconditioner
	$(MAKE) -C CurlLSFEM_example
	$(MAKE) -C DivLSFEM_example

SUBDIRS = Epetra_Basic_Perf Stratimikos_Solver_Driver Stratimikos_Preconditioner CurlLSFEM_example DivLSFEM_example

.PHONY: clean $(SUBDIRS)

clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) clean -C $@

