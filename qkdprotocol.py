
import random
import numpy as np

class InputError(Exception):
    def __int__(self, expression, message):
        self.expression = expression
        self.message = message

class Photon:

    def __init__(self, Hcomp=0, Vcomp=0):
        self.alpha = Hcomp
        self.beta  = Vcomp

    # This is for debugging purposes only!
    def toString(self):
        if np.isreal(self.alpha):
            string = str(self.alpha) + "|H> "
        else:
            string = str(self.alpha) + "|H> "
        if np.isreal(self.beta):
            if self.beta >= 0:
                string += "+ " + str(self.beta) + "|V>"
            else:
                string += "- " + str(-self.beta) + "|V>"
        else:
            string += "+ " + str(self.beta) + "|V>"
        return string

    def prepareVacuum(self):
        energyPerMode = 0.5; # in units of hbar*omega
        x0 = np.sqrt(energyPerMode)*random.gauss(0,1)/np.sqrt(2)
        y0 = np.sqrt(energyPerMode)*random.gauss(0,1)/np.sqrt(2)
        x1 = np.sqrt(energyPerMode)*random.gauss(0,1)/np.sqrt(2)
        y1 = np.sqrt(energyPerMode)*random.gauss(0,1)/np.sqrt(2)
        self.alpha = complex(x0, y0)
        self.beta  = complex(x1, y1)

    def prepare(self, alpha, beta, avgPhotonNumber):
        if avgPhotonNumber < 0:
            raise InputError()
        vac = Photon()
        vac.prepareVacuum()
        self.alpha = alpha * np.sqrt(avgPhotonNumber) + vac.alpha
        self.beta  = beta  * np.sqrt(avgPhotonNumber) + vac.beta

    def prepareH(self, avgPhotonNumber):
        self.prepare(1, 0, avgPhotonNumber)

    def prepareV(self, avgPhotonNumber):
        self.prepare(0,1, avgPhotonNumber)

    def prepareD(self, avgPhotonNumber):
        self.prepare(1/np.sqrt(2),  1/np.sqrt(2), avgPhotonNumber)

    def prepareA(self, avgPhotonNumber):
        self.prepare(1/np.sqrt(2), -1/np.sqrt(2), avgPhotonNumber)

    def prepareR(self, avgPhotonNumber):
        self.prepare(1/np.sqrt(2),  1j/np.sqrt(2), avgPhotonNumber)

    def prepareL(self, avgPhotonNumber):
        self.prepare(1/np.sqrt(2), -1j/np.sqrt(2), avgPhotonNumber)

    def measureHV(self, probDarkCount):
        if probDarkCount < 0 or probDarkCount > 1:
            raise InputError
        threshold  = -0.5*np.log(1 - np.sqrt(1-probDarkCount))
        intensityH = abs(self.alpha)**2
        intensityV = abs(self.beta)**2
        # The photon is absorbed by the detector:
        self.prepareVacuum()
        # The outcome is determined by threshold exceedances:
        if intensityH <= threshold and intensityV <= threshold:
            return "N" # no detection (invalid measurement)
        elif intensityH >  threshold and intensityV <= threshold:
            return "H" # single H photon detected
        elif intensityH <=  threshold and intensityV > threshold:
            return "V" # single V photon detected
        else:
            return "M" # multiple detections (invalid measurement)

    def measureDA(self, probDarkCount):
        a = self.alpha
        b = self.beta
        self.alpha = (a+b)/np.sqrt(2)
        self.beta  = (a-b)/np.sqrt(2)
        outcome = self.measureHV(probDarkCount)
        if outcome == "H": return "D"
        if outcome == "V": return "A"
        else: return outcome

    def measureRL(self, probDarkCount):
        a = self.alpha
        b = self.beta
        self.alpha = (a-b*1j)/np.sqrt(2)
        self.beta  = (a+b*1j)/np.sqrt(2)
        outcome = self.measureHV(probDarkCount)
        if outcome == "H": return "R"
        if outcome == "V": return "L"
        else: return outcome

    def applyPolarizer(self, theta, phi):
	# Apply a polarizing filter according to the input parameters.
	# theta=0,    phi=0:     H polarizer
	# theta=pi/2, phi=0:     V polarizer
	# theta=pi/4, phi=0:     D polarizer
	# theta=pi/4, phi=pi:    A polarizer
	# theta=pi/4, phi=+pi/2: R polarizer
	# theta=pi/4, phi=-pi/2: L polarizer
        z = np.exp(1j*phi)
        a = self.alpha
        b = self.beta
        self.alpha = a*(1+np.cos(2*theta))/2 + b*np.sin(2*theta)/2*np.conj(z)
        self.beta  = a*np.sin(2*theta)/2*z + b*(1-np.cos(2*theta))/2
        # Now add an extra vacuum component.
        vac = photon.Photon()
        vac.prepareVacuum()
        a = vac.alpha
        b = vac.beta
        self.alpha = self.alpha + a*np.sin(theta)**2 + b*(-np.sin(2*theta)/2)*np.conj(z)
        self.beta  = self.beta  + a*(-np.sin(2*theta)/2)*z + b*np.cos(theta)**2

    def applyUnitaryGate(self, theta, phi, lamb):
        U = [[0,0],[0,0]]
        z1 = np.exp(1j*phi)
        z2 = -np.exp(1j*lamb)
        z3 = np.exp(1j*(lamb+phi))
        U[0][0] = np.cos(theta/2)
        U[1][0] = np.sin(theta/2)*z1
        U[0][1] = np.sin(theta/2)*z2
        U[1][1] = np.cos(theta/2)*z3
        a = self.alpha
        b = self.beta
        self.alpha = U[0][0]*a + U[0][1]*b
        self.beta  = U[1][0]*a + U[1][1]*b

    def applyXGate(self):
        # Applies the Pauli X gate
        self.applyUnitaryGate(np.pi, 0, np.pi)

    def applyYGate(self):
        # Applies the Pauli Y gate
        self.applyUnitaryGate(np.pi, np.pi/2, np.pi/2)

    def applyZGate(self):
        # Applies the Pauli X gate
        self.applyUnitaryGate(0, np.pi, 0)

    def applyHGate(self):
        # Applied the Hadamard (half-wavelength) gate
        self.applyUnitaryGate(np.pi/2, 0, np.pi)

    def applyQGate(self):
        # Applies the SH (quarter-wavelength) gate
        self.applyUnitaryGate(np.pi/2, np.pi/2, np.pi)

    def applyNoisyGate(self, p):
        # This operation acts as a depolarizing channel.
	# p = 0 leaves the photon unchanged.
	# p = 1 yields a completely random photon.
	# 0 < p < 1 yields a partially random photon.
        if p < 0 or p > 1:
            raise InputError
        theta = np.arccos(1 - 2*random.uniform(0,1)*p)
        phi   = p*(2*random.uniform(0,1) - 1)*np.pi
        lamb  = p*(2*random.uniform(0,1) - 1)*np.pi
        self.applyUnitaryGate(theta, phi, lamb)

    def applyAttenuation(self, r):
	# This operation acts as a partially reflecting beam splitter.
	# r = 0 leaves the photon unchanged.
	# r = 1 completely absorbs the photon, leaving a vacuum state.
	# 0 < r < 1 partially attenuates the photon and adds some vacuum.
	# r is the reflectivity.
        if r < 0 or r > 1:
            raise InputError
        t = np.sqrt(1-r*r) # t is the transmissivity.
        vac = photon.Photon()
        vac.prepareVacuum()
        self.alpha = (self.alpha)*t + (vac.alpha)*r
        self.beta  = (self.beta )*t + (vac.beta)*r



n = 100 # number of photons



# Alice --------------------------------------------

# Alice generates the raw key.
keyAlice = ""
for i in range(n): # Iterate over the number of photons.
    if random.randint(0,1)==0: # Flip a coin (0 or 1).
        keyAlice += '0'
    else:
        keyAlice += '1'

# Alice chooses the encoding basis for each key bit.
basisAlice = ""
for i in range(n):
  if random.randint(0,1)==0:
    basisAlice += '+'
  else:
    basisAlice += 'x'

# Alice selects a photon state according to the key and basis.
photonAlice = ""
for i in range(n):
  if basisAlice[i] == '+':
    if keyAlice[i] == '0':
      photonAlice += 'H'
    else:
      photonAlice += 'V'
  else:
    if keyAlice[i] == '0':
      photonAlice += 'D'
    else:
      photonAlice += 'A'

# Alice prepares and sends each photon.
photonArray = [Photon() for i in range(n)]
for i in range(n):
  if photonAlice[i] == 'H':
    photonArray[i].prepareH(1)
  elif photonAlice[i] == 'V':
    photonArray[i].prepareV(1)
  elif photonAlice[i] == 'D':
    photonArray[i].prepareD(1)
  else:
    photonArray[i].prepareA(1)



# Eve   --------------------------------------------

# Eve selects a subsample of photons from Alice to measure.
interceptIndex = ""
for i in range(n):
  if random.randint(0,1)==0:
    interceptIndex += '0'
  else:
    interceptIndex += '1'

# Eve chooses a basis to measure each intercepted photon.
basisEve = ""
for i in range(n):
  if interceptIndex[i] == '1':
    if random.randint(0,1)==0:
      basisEve += '+'
    else:
      basisEve += 'x'
  else:
    basisEve += ' '

# Eve performs a measurement on each photon.
outcomeEve = ""
for i in range(n):
  if basisEve[i] == '+':
    outcomeEve += photonArray[i].measureHV(.5)
  elif basisEve[i] == 'x':
    outcomeEve += photonArray[i].measureDA(.5)
  else:
    outcomeEve += ' '

# Eve resends photons to Bob.
for i in range(n):
    if outcomeEve[i] == 'H':
      photonArray[i].prepareH(1)
    elif outcomeEve[i] == 'V':
      photonArray[i].prepareV(1)
    elif outcomeEve[i] == 'D':
      photonArray[i].prepareD(1)
    elif outcomeEve[i] == 'A':
      photonArray[i].prepareA(1)


# Bob   --------------------------------------------

# Bob chooses a basis to measure each photon.
basisBob = ""
for i in range(n):
  if random.randint(0,1)==0:
    basisBob += '+'
  else:
    basisBob += 'x'

# Bob performs a measurement on each photon.
outcomeBob = ""
for i in range(n):
  if basisBob[i] == '+':
    outcomeBob += photonArray[i].measureHV(.5)
  else:
    outcomeBob += photonArray[i].measureDA(.5)

# Bob infers the raw key.
keyBob = ""
for i in range(n):
  if outcomeBob[i] == 'H':
    keyBob += '0'
  elif outcomeBob[i] == 'V':
    keyBob += '1'
  elif outcomeBob[i] == 'D':
    keyBob += '0'
  elif outcomeBob[i] == 'A':
    keyBob += '1'
  else:
    keyBob += '-'



# -----------------------------------------------------------
# Alice and Bob now publicly announce which bases they chose.
# Bob also announces which of his measurements were invalid.
# -----------------------------------------------------------

# Alice & Bob ----------------------------------------------------------

# Alice and Bob extract their sifted keys.
siftedAlice = ""
siftedBob   = ""
for i in range(n):
  if basisAlice[i] == basisBob[i] and keyBob[i] != '-':
    siftedAlice += keyAlice[i]
    siftedBob += keyBob[i]
  else:
    siftedAlice += ' '
    siftedBob += ' '

# Alice and Bob use a portion of their sifted keys to estimate the quantum bit error rate (QBER).
sampleIndex = ""
sampledBobQBER = 0
numOnes = 0
for i in range(n):
  if random.randint(0,1)==0:
    sampleIndex += '0'
  else:
    sampleIndex += '1'
    if siftedAlice[i] != ' ':
      numOnes += 1
numMatches = 0
for i in range(n):
  if sampleIndex[i] == '1' and siftedAlice[i] == siftedBob[i] and siftedAlice[i] != ' ':
      numMatches += 1
sampledBobQBER = 1 - (numMatches / numOnes)

# Alice and Bob remove the portion of their sifted keys that was sampled.
secureAlice = ""
secureBob = ""
for i in range(n):
  if sampleIndex[i] == '1':
    secureAlice += ' '
    secureBob += ' '
  else:
    secureAlice += siftedAlice[i]
    secureBob += siftedBob[i]

# Alice and Bob make a hard determination whether the channel is secure.
channelSecure = True # default value, to be changed to False if Eve suspected
if sampledBobQBER >= .5:
  channelSecure = False



# Eve ------------------------------------------------------------------

# Eve infers the raw key.
keyEve = ""
for i in range(n):
  if outcomeEve[i] == 'H':
    keyEve += '0'
  elif outcomeEve[i] == 'V':
    keyEve += '1'
  elif outcomeEve[i] == 'D':
    keyEve += '0'
  elif outcomeEve[i] == 'A':
    keyEve += '1'
  elif outcomeEve[i] == ' ':
    keyEve += ' '
  else:
    keyEve += '-'

# Eve extracts her sifted key.
stolenEve = ""
for i in range(n):
  if basisAlice[i] != basisEve[i]:
    stolenEve += ' '
  elif sampleIndex[i] == '1':
    stolenEve += ' '
  elif keyEve[i] == '-':
    stolenEve += ' '
  else:
    stolenEve += keyEve[i]



# ANALYSIS -------------------------------------------------------------

# Compare Alice and Bob's sifted keys.
numMatchBob = 0
actualBobQBER = 0
secureKeyRateBob = 0
secureKeyLengthBob = 0
for i in range(len(secureAlice)):
    if secureAlice[i] != ' ':
       secureKeyLengthBob += 1
       if secureAlice[i] == secureBob[i]:
           numMatchBob += 1

# Compute the actual quantum bit error rate for Bob.
if secureKeyLengthBob > 0:
    actualBobQBER = 1 - numMatchBob / secureKeyLengthBob
else:
    actualBobQBER = float('nan')
# Compute the secure key rate, assuming each trial takes 1 microsecond.
secureKeyRateBob = (1-actualBobQBER) * secureKeyLengthBob / n * 1e6;

# Compare Alice and Eve's sifted keys.
numMatchEve = 0
actualEveQBER = 0
stolenKeyRateEve = 0
stolenKeyLengthEve = 0
for i in range(len(stolenEve)):
    if stolenEve[i] != ' ':
       stolenKeyLengthEve += 1
       if secureAlice[i] == stolenEve[i]:
           numMatchEve += 1
# Compute the actual quantum bit error rate for Eve.
if stolenKeyLengthEve > 0:
    actualEveQBER = 1 - numMatchEve / stolenKeyLengthEve
else:
    actualEveQBER = float('nan')
# Compute the stolen key rate, assuming each trial takes 1 microsecond.
stolenKeyRateEve = (1-actualEveQBER) * stolenKeyLengthEve / n * 1e6;


# DISPLAY RESULTS ------------------------------------------------------

print("")
print("basisAlice  = " + basisAlice)
print("basisBob    = " + basisBob)
print("basisEve    = " + basisEve)
print("")
print("keyAlice    = " + keyAlice)
print("keyBob      = " + keyBob)
print("keyEve      = " + keyEve)
print("")
print("siftedAlice = " + siftedAlice)
print("siftedBob   = " + siftedBob)
print("")
print("secureAlice = " + secureAlice)
print("secureBob   = " + secureBob)
print("stolenEve   = " + stolenEve)
print("")
if not channelSecure:
    secureKeyRateBob = 0;
    stolenKeyRateEve = 0;
    print("*********************************************")
    print("* ALERT! The quantum channel is not secure. *")
    print("*********************************************")
    print("")
print("sampledBobQBER = " + str(sampledBobQBER))
print("actualBobQBER  = " + str(actualBobQBER))
print("actualEveQBER  = " + str(actualEveQBER))
print("")
print("secureKeyRateBob = " + str(secureKeyRateBob/1000) + " kbps")
print("stolenKeyRateEve = " + str(stolenKeyRateEve/1000) + " kbps")

# Your goal is to maximize secureKeyRateBob and minimize stolenKeyRateEve.