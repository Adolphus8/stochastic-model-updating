## Case Study - TRIGA Nuclear Reactor

source("pba BETTER.r")
install.packages("tictoc")
library(tictoc)

#######################################################################
readkeygraph <- function(prompt) {
getGraphicsEvent(prompt=prompt,onMouseDown=NULL,onMouseMove=NULL,onMouseUp=NULL,onKeybd=onKeybd,consolePrompt="[click on graph then follow top prompt to continue]")
Sys.sleep(0.01)
return(keyPressed)}

onKeybd <- function(key) keyPressed <<- key
wait = function(msg='') invisible(readkeygraph(paste(msg,'[press any key to continue]')))
rbyc = function(r=1,c=1) par(mfrow=c(r,c))
pl = function(..., xlab='',ylab='Cumulative probability',ylim=c(0,1)) plot(NULL,ylim=ylim,xlim=range(...),xlab=xlab,ylab=ylab)
sh2 = function(w, wi=NULL, wii=NULL, m, pl=range(w)) {plot(w,xlim=pl); if (!missing(wi)) red(wii); green(wi); blue(w); title(m)}
#######################################################################

Pbox$steps = 1000; samp = 1000;

# Import the sample data to construct the distribution-free P-boxes:
# Construct P-boxes for Basic events: 1, 3, 5, 6, 7, 8, 9, 10, 11
raw_data = read.csv("pbox_R1_R3.csv")
pbox1 = env(pbox.samples(raw_data[1:nrow(raw_data), 1]), pbox.samples(raw_data[1:nrow(raw_data), 2])); # Basic event 1: Valve V-3 failed to open 
pbox3 = env(pbox.samples(raw_data[1:nrow(raw_data), 1]), pbox.samples(raw_data[1:nrow(raw_data), 2])); # Basic event 3: Valve V-4 failed to open

# Define the component C-boxes for Basic events: 2, 4, 5:12
cbox2  = KN(345, 100000);  # Basic event 2: Operator failed to open valve V-3
cbox4  = KN(345, 100000);  # Basic event 4: Operator failed to open valve V-4
cbox5  = KN(1, 42);        # Basic event 5: Inlet pipe of V-3 breaks
cbox6  = KN(1, 42);        # Basic event 6: Inlet pipe of V-4 breaks
cbox7  = KN(0, 42);        # Basic event 7: Thermal column breaks
cbox8  = KN(1, 42);        # Basic event 8: Radial beam port 1 breaks
cbox9  = KN(1, 42);        # Basic event 9: Radial beam port 2 breaks
cbox10  = KN(0, 42);       # Basic event 10: Tangential beam port 1 breaks
cbox11  = KN(0, 42);       # Basic event 11: Tangential beam port 2 breaks
cbox12  = KN(1, 42);       # Basic event 12: Reactor pool breaks

tic("Propagation under total independence")
# Define the Event C-boxes under independence:
e4ai = orI(pbox1, cbox2);                        cbox_vec = array(c(e4ai@d, e4ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E4a_cbox_indep_TRIGA.csv")
e4bi = orI(pbox3, cbox4);                        cbox_vec = array(c(e4bi@d, e4bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E4b_cbox_indep_TRIGA.csv")
e4ci = orI(cbox8, cbox9);                        cbox_vec = array(c(e4ci@d, e4ci@u),dim = c(samp,2)); write.csv(cbox_vec, "E4c_cbox_indep_TRIGA.csv")
e4di = orI(cbox10, cbox11);                      cbox_vec = array(c(e4di@d, e4di@u),dim = c(samp,2)); write.csv(cbox_vec, "E4d_cbox_indep_TRIGA.csv")
e3ai = orI(e4ai, e4bi);                          cbox_vec = array(c(e3ai@d, e3ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_cbox_indep_TRIGA.csv")
e3bi = orI(cbox5, cbox6);                        cbox_vec = array(c(e3bi@d, e3bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_cbox_indep_TRIGA.csv")
e3ci = orI(e4ci, e4di);                          cbox_vec = array(c(e3ci@d, e3ci@u),dim = c(samp,2)); write.csv(cbox_vec, "E3c_cbox_indep_TRIGA.csv")
e2ai = andI(e3ai, e3bi);                         cbox_vec = array(c(e2ai@d, e2ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_cbox_indep_TRIGA.csv")
e2bi = orI(orI(cbox7, cbox12), e3ci);            cbox_vec = array(c(e2bi@d, e2bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_cbox_indep_TRIGA.csv")
e1i = orI(e2ai, e2bi);                           cbox_vec = array(c(e1i@d, e1i@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_cbox_indep_TRIGA.csv")
toc()

tic("Propagation under uncertain dependence")
# Define the Event C-boxes under uncertain dependence:
e4af = or(pbox1, cbox2);                      cbox_vec = array(c(e4af@d, e4af@u),dim = c(samp,2)); write.csv(cbox_vec, "E4a_cbox_frechet_TRIGA.csv")
e4bf = or(pbox3, cbox4);                      cbox_vec = array(c(e4bf@d, e4bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E4b_cbox_frechet_TRIGA.csv")
e4cf = or(cbox8, cbox9);                      cbox_vec = array(c(e4cf@d, e4cf@u),dim = c(samp,2)); write.csv(cbox_vec, "E4c_cbox_frechet_TRIGA.csv")
e4df = or(cbox10, cbox11);                    cbox_vec = array(c(e4df@d, e4df@u),dim = c(samp,2)); write.csv(cbox_vec, "E4d_cbox_frechet_TRIGA.csv")
e3af = or(e4af, e4bf);                        cbox_vec = array(c(e3af@d, e3af@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_cbox_frechet_TRIGA.csv")
e3bf = or(cbox5, cbox6);                      cbox_vec = array(c(e3bf@d, e3bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_cbox_frechet_TRIGA.csv")
e3cf = orI(e4cf, e4df);                       cbox_vec = array(c(e3cf@d, e3cf@u),dim = c(samp,2)); write.csv(cbox_vec, "E3c_cbox_frechet_TRIGA.csv")
e2af = and(e3af, e3bf);                       cbox_vec = array(c(e2af@d, e2af@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_cbox_frechet_TRIGA.csv")
e2bf = or(or(cbox7, cbox12), e3cf);           cbox_vec = array(c(e2bf@d, e2bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_cbox_frechet_TRIGA.csv")
e1f = or(e2af, e2bf);                         cbox_vec = array(c(e1f@d, e1f@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_cbox_frechet_TRIGA.csv")
toc()

tic("Propagation under reasonable independence")
# Define the Event C-boxes under independence:
e4a = orI(pbox1, cbox2);                     cbox_vec = array(c(e4a@d, e4a@u),dim = c(samp,2)); write.csv(cbox_vec, "E4a_cbox_ri_TRIGA.csv")
e4b = orI(pbox3, cbox4);                     cbox_vec = array(c(e4b@d, e4b@u),dim = c(samp,2)); write.csv(cbox_vec, "E4b_cbox_ri_TRIGA.csv")
e4c = or(cbox8, cbox9);                      cbox_vec = array(c(e4c@d, e4c@u),dim = c(samp,2)); write.csv(cbox_vec, "E4c_cbox_ri_TRIGA.csv")
e4d = or(cbox10, cbox11);                    cbox_vec = array(c(e4d@d, e4d@u),dim = c(samp,2)); write.csv(cbox_vec, "E4d_cbox_ri_TRIGA.csv")
e3a = or(e4a, e4b);                          cbox_vec = array(c(e3a@d, e3a@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_cbox_ri_TRIGA.csv")
e3b = orI(cbox5, cbox6);                     cbox_vec = array(c(e3b@d, e3b@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_cbox_ri_TRIGA.csv")
e3c = orI(e4c, e4d);                         cbox_vec = array(c(e3c@d, e3c@u),dim = c(samp,2)); write.csv(cbox_vec, "E3c_cbox_ri_TRIGA.csv")
e2a = and(e3a, e3b);                         cbox_vec = array(c(e2a@d, e2a@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_cbox_ri_TRIGA.csv")
e2b = orI(orI(cbox7, cbox12), e3c);          cbox_vec = array(c(e2b@d, e2b@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_cbox_ri_TRIGA.csv")
e1 = orI(e2a, e2b);                          cbox_vec = array(c(e1@d, e1@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_cbox_ri_TRIGA.csv")
toc()

# Plot the C-boxes of the Events
rbyc(4,3)
sh2(e1f, e1i, e1, 'E1 (Top event)', c(0, 1))
sh2(e2af, e2ai, e2a, 'E2a', c(0, 0.05))
sh2(e2bf, e2bi, e2b, 'E2b', c(0, 1))
sh2(e3af, e3ai, e3a, 'E3a', c(0, 0.05))
sh2(e3bf, e3bi, e3b, 'E3b', c(0, 0.4))
sh2(e3cf, e3ci, e3c, 'E3c', c(0, 0.4))
sh2(e4af, e4ai, e4a, 'E4a', c(0, 0.025))
sh2(e4bf, e4bi, e4b, 'E4b', c(0, 0.025))
sh2(e4cf, e4ci, e4c, 'E4c', c(0, 0.53))
sh2(e4df, e4di, e4d, 'E4d', c(0, 0.25))
wait()

## Results of the 95% two-sided symmetrical confidence intervals:
conf_lb = 0.025; conf_ub = 1 - conf_lb;

# Under total independence:
interval_e1i = interval(cut(e1i, conf_lb), cut(e1i, conf_ub))
interval_e2ai = interval(cut(e2ai, conf_lb), cut(e2ai, conf_ub))
interval_e2bi = interval(cut(e2bi, conf_lb), cut(e2bi, conf_ub))
interval_e3ai = interval(cut(e3ai, conf_lb), cut(e3ai, conf_ub))
interval_e3bi = interval(cut(e3bi, conf_lb), cut(e3bi, conf_ub))
interval_e3ci = interval(cut(e3ci, conf_lb), cut(e3ci, conf_ub))
interval_e4ai = interval(cut(e4ai, conf_lb), cut(e4ai, conf_ub))
interval_e4bi = interval(cut(e4bi, conf_lb), cut(e4bi, conf_ub))
interval_e4ci = interval(cut(e4ci, conf_lb), cut(e4ci, conf_ub))
interval_e4di = interval(cut(e4di, conf_lb), cut(e4di, conf_ub))

# Under uncertain dependence:
interval_e1f = interval(cut(e1f, conf_lb), cut(e1f, conf_ub))
interval_e2af = interval(cut(e2af, conf_lb), cut(e2af, conf_ub))
interval_e2bf = interval(cut(e2bf, conf_lb), cut(e2bf, conf_ub))
interval_e3af = interval(cut(e3af, conf_lb), cut(e3af, conf_ub))
interval_e3bf = interval(cut(e3bf, conf_lb), cut(e3bf, conf_ub))
interval_e3cf = interval(cut(e3cf, conf_lb), cut(e3cf, conf_ub))
interval_e4af = interval(cut(e4af, conf_lb), cut(e4af, conf_ub))
interval_e4bf = interval(cut(e4bf, conf_lb), cut(e4bf, conf_ub))
interval_e4cf = interval(cut(e4cf, conf_lb), cut(e4cf, conf_ub))
interval_e4df = interval(cut(e4df, conf_lb), cut(e4df, conf_ub))

# Under reasonable independence:
interval_e1 = interval(cut(e1, conf_lb), cut(e1, conf_ub))
interval_e2a = interval(cut(e2a, conf_lb), cut(e2a, conf_ub))
interval_e2b = interval(cut(e2b, conf_lb), cut(e2b, conf_ub))
interval_e3a = interval(cut(e3a, conf_lb), cut(e3a, conf_ub))
interval_e3b = interval(cut(e3b, conf_lb), cut(e3b, conf_ub))
interval_e3c = interval(cut(e3c, conf_lb), cut(e3c, conf_ub))
interval_e4a = interval(cut(e4a, conf_lb), cut(e4a, conf_ub))
interval_e4b = interval(cut(e4b, conf_lb), cut(e4b, conf_ub))
interval_e4c = interval(cut(e4c, conf_lb), cut(e4c, conf_ub))
interval_e4d = interval(cut(e4d, conf_lb), cut(e4d, conf_ub))


