source("pba BETTER.r")

#######################################################################
readkeygraph <- function(prompt) {
getGraphicsEvent(prompt=prompt,onMouseDown=NULL,onMouseMove=NULL,onMouseUp=NULL,onKeybd=onKeybd,consolePrompt="[click on graph then follow top prompt to continue]")
Sys.sleep(0.01)
return(keyPressed)}

onKeybd <- function(key) keyPressed <<- key
wait = function(msg='') invisible(readkeygraph(paste(msg,'[press any key to continue]')))
rbyc = function(r=1,c=1) par(mfrow=c(r,c))
pl = function(..., xlab='',ylab='Cumulative probability',ylim=c(0,1)) plot(NULL,ylim=ylim,xlim=range(...),xlab=xlab,ylab=ylab)
sh = function(w, wi=NULL, m, pl=range(w)) {plot(w,xlim=pl); if (!missing(wi)) green(wi); blue(w); title(m)}
#######################################################################

# Define the component C-boxes:
samp = 1000; Pbox$steps = samp;
R1 = beta(interval(0.9920, 1.9499), interval(32.7640, 49.0391));
R2 = makepbox(3.45e-05);
R3 = beta(interval(1.3256, 2.2276), interval(36.7131, 54.2610));
R4 = makepbox(3.45e-05);
R5 = beta(interval(1.2825, 1.8426), interval(43.7148, 72.9314));
R6 = beta(interval(1.1291, 1.9603), interval(54.0285, 72.9051));

# Plot the C-boxes of the components:
rbyc(2,3)
sh(R1,,'Valve V-3',c(0, 0.4))
sh(R2,,'Operator of V-3',c(2e-5, 5e-5))
sh(R3,,'Valve V-4',c(0, 0.4))
sh(R4,,'Operator of V-4',c(2e-5, 5e-5))
sh(R5,,'Inlet pipe of V-3',c(0, 0.4))
sh(R6,,'Inlet pipe of V-4',c(0, 0.4))
wait()

# Define the Event C-boxes under uncertain dependencies:
e3af = or(R1, R2);     cbox_vec = array(c(e3af@d, e3af@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_pbox_frechet_TRIGA.csv")  
e3bf = or(R3, R4);     cbox_vec = array(c(e3bf@d, e3bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_pbox_frechet_TRIGA.csv")  
e2af = or(e3af, e3bf); cbox_vec = array(c(e2af@d, e2af@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_pbox_frechet_TRIGA.csv")  
e2bf = or(R5, R6);     cbox_vec = array(c(e2bf@d, e2bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_pbox_frechet_TRIGA.csv")  
e1f = and(e2af, e2bf); cbox_vec = array(c(e1f@d, e1f@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_pbox_frechet_TRIGA.csv")  

# Define the Event C-boxes under independence:
e3ai = orI(R1, R2);     cbox_vec = array(c(e3ai@d, e3ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_pbox_indep_TRIGA.csv") 
e3bi = orI(R3, R4);     cbox_vec = array(c(e3bi@d, e3bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_pbox_indep_TRIGA.csv") 
e2ai = orI(e3ai, e3bi); cbox_vec = array(c(e2ai@d, e2ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_pbox_indep_TRIGA.csv") 
e2bi = orI(R5, R6);     cbox_vec = array(c(e2bi@d, e2bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_pbox_indep_TRIGA.csv") 
e1i = andI(e2ai, e2bi); cbox_vec = array(c(e1i@d, e1i@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_pbox_indep_TRIGA.csv") 


# Plot the C-boxes of the Events
rbyc(2,3)
sh(e1f,e1i,'E1',c(0,1))
sh(e2af,e2ai,'E2a',c(0,1))
sh(e2bf,e2bi,'E2b',c(0,1))
sh(e3af,e3ai,'E3a',c(0,1))
sh(e3bf,e3bi,'E3b',c(0,1))
wait()

################################################################################################################
################################################################################################################

## Perform the computation with the true solution:

# Define the component C-boxes:
R1 = beta(1.50, 43.00);
R2 = makepbox(3.45e-05);
R3 = beta(1.50, 43.00);
R4 = makepbox(3.45e-05);
R5 = beta(1.50, 62.40);
R6 = beta(1.50, 62.40);

# Plot the C-boxes of the components:
rbyc(2,3)
sh(R1,,'Valve V-3',c(0, 0.4))
sh(R2,,'Operator of V-3',c(2e-5, 5e-5))
sh(R3,,'Valve V-4',c(0, 0.4))
sh(R4,,'Operator of V-4',c(2e-5, 5e-5))
sh(R5,,'Inlet pipe of V-3',c(0, 0.4))
sh(R6,,'Inlet pipe of V-4',c(0, 0.4))
wait()

# Define the Event C-boxes under uncertain dependencies:
e3af = or(R1, R2);     cbox_vec = array(c(e3af@d, e3af@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_beta_frechet_TRIGA.csv")  
e3bf = or(R3, R4);     cbox_vec = array(c(e3bf@d, e3bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_beta_frechet_TRIGA.csv")  
e2af = or(e3af, e3bf); cbox_vec = array(c(e2af@d, e2af@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_beta_frechet_TRIGA.csv")  
e2bf = or(R5, R6);     cbox_vec = array(c(e2bf@d, e2bf@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_beta_frechet_TRIGA.csv")  
e1f = and(e2af, e2bf); cbox_vec = array(c(e1f@d, e1f@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_beta_frechet_TRIGA.csv")  

# Define the Event C-boxes under independence:
e3ai = orI(R1, R2);     cbox_vec = array(c(e3ai@d, e3ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E3a_beta_indep_TRIGA.csv") 
e3bi = orI(R3, R4);     cbox_vec = array(c(e3bi@d, e3bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E3b_beta_indep_TRIGA.csv") 
e2ai = orI(e3ai, e3bi); cbox_vec = array(c(e2ai@d, e2ai@u),dim = c(samp,2)); write.csv(cbox_vec, "E2a_beta_indep_TRIGA.csv") 
e2bi = orI(R5, R6);     cbox_vec = array(c(e2bi@d, e2bi@u),dim = c(samp,2)); write.csv(cbox_vec, "E2b_beta_indep_TRIGA.csv") 
e1i = andI(e2ai, e2bi); cbox_vec = array(c(e1i@d, e1i@u),dim = c(samp,2)); write.csv(cbox_vec, "E1_beta_indep_TRIGA.csv") 


# Plot the C-boxes of the Events
rbyc(2,3)
sh(e1f,e1i,'E1',c(0,1))
sh(e2af,e2ai,'E2a',c(0,1))
sh(e2bf,e2bi,'E2b',c(0,1))
sh(e3af,e3ai,'E3a',c(0,1))
sh(e3bf,e3bi,'E3b',c(0,1))
wait()