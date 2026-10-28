/**
 * Oceanic Cruise Line — Passenger Safety & Survival Predictor Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Identity
  const inputName = document.getElementById('input-name');
  const inputTitle = document.getElementById('input-title');
  const inputGender = document.getElementById('input-gender');
  const genderMaleBtn = document.getElementById('gender-male');
  const genderFemaleBtn = document.getElementById('gender-female');
  const inputAge = document.getElementById('input-age');
  const ageDisplay = document.getElementById('age-display');
  const ageTierDisplay = document.getElementById('age-tier-display');

  // Accommodations
  const inputPclass = document.getElementById('input-pclass');
  const classCards = document.querySelectorAll('.class-card');
  const inputFare = document.getElementById('input-fare');
  const fareDisplay = document.getElementById('fare-display');
  const inputEmbarked = document.getElementById('input-embarked');
  const portChips = document.querySelectorAll('.port-chip');

  // Companionship
  const inputSibsp = document.getElementById('input-sibsp');
  const sibspVal = document.getElementById('sibsp-val');
  const sibspMinus = document.getElementById('sibsp-minus');
  const sibspPlus = document.getElementById('sibsp-plus');

  const inputParch = document.getElementById('input-parch');
  const parchVal = document.getElementById('parch-val');
  const parchMinus = document.getElementById('parch-minus');
  const parchPlus = document.getElementById('parch-plus');

  const summaryFamilySize = document.getElementById('summary-family-size');
  const summaryAlone = document.getElementById('summary-alone');

  // Form & Buttons
  const passengerForm = document.getElementById('passenger-form');
  const btnSubmit = document.getElementById('btn-submit');
  const btnReset = document.getElementById('btn-reset');
  const presetsContainer = document.getElementById('presets-container');

  // Results Outputs
  const gaugeCircle = document.getElementById('gauge-circle');
  const gaugePercent = document.getElementById('gauge-percent');
  const statusBanner = document.getElementById('status-banner');
  const bannerIcon = document.getElementById('banner-icon');
  const bannerTitle = document.getElementById('banner-title');
  const bannerDesc = document.getElementById('banner-desc');
  const resultsTime = document.getElementById('results-time');

  const lrScoreDisplay = document.getElementById('lr-score-display');
  const lrProgressBar = document.getElementById('lr-progress-bar');
  const rfScoreDisplay = document.getElementById('rf-score-display');
  const rfProgressBar = document.getElementById('rf-progress-bar');

  const factorsList = document.getElementById('factors-list');

  // Boarding Pass Elements
  const passTicket = document.getElementById('pass-ticket');
  const passName = document.getElementById('pass-name');
  const passClass = document.getElementById('pass-class');
  const passPort = document.getElementById('pass-port');
  const passAgeGender = document.getElementById('pass-age-gender');
  const passParty = document.getElementById('pass-party');
  const passStatusSeal = document.getElementById('pass-status-seal');
  const passSealText = document.getElementById('pass-seal-text');
  const passSealRate = document.getElementById('pass-seal-rate');

  // Circumference of SVG circle (r = 68 -> 2 * PI * 68 ≈ 427.26)
  const GAUGE_CIRCUMFERENCE = 427.26;

  // -------------------------------------------------------------------------
  // 1. Initial Setup & Presets Loading
  // -------------------------------------------------------------------------
  async function loadPresets() {
    try {
      const res = await fetch('/api/presets');
      if (!res.ok) throw new Error('Failed to load presets');
      const presets = await res.json();
      renderPresets(presets);
    } catch (err) {
      console.error(err);
      presetsContainer.innerHTML = '<div class="preset-error">Could not load presets</div>';
    }
  }

  function renderPresets(presets) {
    presetsContainer.innerHTML = '';
    presets.forEach((preset, idx) => {
      const chip = document.createElement('div');
      chip.className = `preset-chip ${idx === 0 ? 'active' : ''}`;
      chip.setAttribute('data-preset-id', preset.id);

      const classLabel = preset.pclass === 1 ? '1st Class' : preset.pclass === 2 ? '2nd Class' : '3rd Class';
      const badgeClass = `class-${preset.pclass}`;

      chip.innerHTML = `
        <div class="preset-name">${preset.label}</div>
        <div class="preset-role">${preset.subtitle}</div>
        <div class="preset-tags">
          <span class="preset-badge ${badgeClass}">${classLabel}</span>
          <span class="preset-badge" style="background: rgba(255,255,255,0.08); color: #94a3b8;">${preset.age} yrs</span>
          <span class="preset-badge" style="background: rgba(56,189,248,0.12); color: #38bdf8;">${preset.sex.toUpperCase()}</span>
        </div>
      `;

      chip.addEventListener('click', () => {
        document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        applyPreset(preset);
      });

      presetsContainer.appendChild(chip);
    });
  }

  function applyPreset(p) {
    inputName.value = p.name;
    inputTitle.value = p.title || 'Auto';
    setGender(p.sex);
    inputAge.value = p.age;
    updateAgeDisplay();
    setPclass(p.pclass);
    inputFare.value = p.fare;
    updateFareDisplay();
    setEmbarked(p.embarked);
    setSibsp(p.sibsp);
    setParch(p.parch);

    // Trigger calculation
    runPrediction();
  }

  // -------------------------------------------------------------------------
  // 2. Interactive Input Controls
  // -------------------------------------------------------------------------

  // Gender Control
  function setGender(gender) {
    inputGender.value = gender;
    if (gender === 'male') {
      genderMaleBtn.classList.add('active');
      genderFemaleBtn.classList.remove('active');
    } else {
      genderFemaleBtn.classList.add('active');
      genderMaleBtn.classList.remove('active');
    }
  }

  genderMaleBtn.addEventListener('click', () => setGender('male'));
  genderFemaleBtn.addEventListener('click', () => setGender('female'));

  // Age Slider & Bracket
  function updateAgeDisplay() {
    const age = parseInt(inputAge.value, 10);
    ageDisplay.textContent = age;
    let tier = 'Adult';
    if (age <= 2) tier = 'Infant';
    else if (age <= 12) tier = 'Child';
    else if (age <= 18) tier = 'Youth';
    else if (age <= 55) tier = 'Adult';
    else tier = 'Senior';
    ageTierDisplay.textContent = tier;
  }
  inputAge.addEventListener('input', updateAgeDisplay);

  // Cabin Class Selection
  function setPclass(pclass) {
    inputPclass.value = pclass;
    classCards.forEach(card => {
      const c = parseInt(card.getAttribute('data-class'), 10);
      if (c === pclass) {
        card.classList.add('active');
      } else {
        card.classList.remove('active');
      }
    });
  }

  classCards.forEach(card => {
    card.addEventListener('click', () => {
      const c = parseInt(card.getAttribute('data-class'), 10);
      setPclass(c);
      // Auto adjust suggested fare if user hasn't explicitly customized
      if (c === 1 && parseFloat(inputFare.value) < 40) {
        inputFare.value = 85;
      } else if (c === 2 && (parseFloat(inputFare.value) > 60 || parseFloat(inputFare.value) < 12)) {
        inputFare.value = 24;
      } else if (c === 3 && parseFloat(inputFare.value) > 25) {
        inputFare.value = 8;
      }
      updateFareDisplay();
    });
  });

  // Fare Slider
  function updateFareDisplay() {
    fareDisplay.textContent = parseFloat(inputFare.value).toFixed(2);
  }
  inputFare.addEventListener('input', updateFareDisplay);

  // Embarked Port Chips
  function setEmbarked(port) {
    inputEmbarked.value = port;
    portChips.forEach(chip => {
      if (chip.getAttribute('data-port') === port) {
        chip.classList.add('active');
      } else {
        chip.classList.remove('active');
      }
    });
  }

  portChips.forEach(chip => {
    chip.addEventListener('click', () => {
      setEmbarked(chip.getAttribute('data-port'));
    });
  });

  // Companionship Steppers
  function updateCompanionship() {
    const sibsp = parseInt(inputSibsp.value, 10);
    const parch = parseInt(inputParch.value, 10);
    const totalFamily = sibsp + parch + 1;
    const isAlone = totalFamily === 1;

    summaryFamilySize.innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
      Total Family Size: <strong>${totalFamily}</strong>
    `;

    summaryAlone.innerHTML = isAlone
      ? `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg> Traveling: <strong>Solo (Alone)</strong>`
      : `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg> Traveling: <strong>With Companions (${totalFamily})</strong>`;
  }

  function setSibsp(val) {
    const clean = Math.max(0, Math.min(8, val));
    inputSibsp.value = clean;
    sibspVal.textContent = clean;
    updateCompanionship();
  }

  function setParch(val) {
    const clean = Math.max(0, Math.min(8, val));
    inputParch.value = clean;
    parchVal.textContent = clean;
    updateCompanionship();
  }

  sibspMinus.addEventListener('click', () => setSibsp(parseInt(inputSibsp.value, 10) - 1));
  sibspPlus.addEventListener('click', () => setSibsp(parseInt(inputSibsp.value, 10) + 1));
  parchMinus.addEventListener('click', () => setParch(parseInt(inputParch.value, 10) - 1));
  parchPlus.addEventListener('click', () => setParch(parseInt(inputParch.value, 10) + 1));

  // Reset Button
  btnReset.addEventListener('click', () => {
    inputName.value = 'Eleanor Margaret Vance';
    inputTitle.value = 'Mrs';
    setGender('female');
    inputAge.value = 32;
    updateAgeDisplay();
    setPclass(1);
    inputFare.value = 85;
    updateFareDisplay();
    setEmbarked('C');
    setSibsp(1);
    setParch(0);
    document.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
    runPrediction();
  });

  // -------------------------------------------------------------------------
  // 3. API Communication & Results Rendering
  // -------------------------------------------------------------------------
  passengerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    runPrediction();
  });

  async function runPrediction() {
    // Collect payload
    const payload = {
      name: inputName.value.trim() || 'Passenger',
      title: inputTitle.value,
      sex: inputGender.value,
      pclass: parseInt(inputPclass.value, 10),
      age: parseFloat(inputAge.value),
      fare: parseFloat(inputFare.value),
      embarked: inputEmbarked.value,
      sibsp: parseInt(inputSibsp.value, 10),
      parch: parseInt(inputParch.value, 10)
    };

    // Button visual loading state
    btnSubmit.disabled = true;
    btnSubmit.querySelector('.btn-text').textContent = 'Analyzing Historical Cruise Data...';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Unable to process prediction. Please check server status.');
    } finally {
      btnSubmit.disabled = false;
      btnSubmit.querySelector('.btn-text').textContent = 'Compute Cruise Survival Probability';
    }
  }

  function renderResults(data) {
    const pred = data.prediction;
    const pass = data.passenger;
    const pct = pred.probability_percent;
    const theme = pred.status_theme;

    // 1. Update Gauge
    const offset = GAUGE_CIRCUMFERENCE - (GAUGE_CIRCUMFERENCE * (pct / 100));
    gaugeCircle.style.strokeDashoffset = offset;

    // Gauge stroke color
    if (theme === 'safe') {
      gaugeCircle.style.stroke = '#10b981';
    } else if (theme === 'warning') {
      gaugeCircle.style.stroke = '#f59e0b';
    } else {
      gaugeCircle.style.stroke = '#f43f5e';
    }

    gaugePercent.textContent = `${pct}%`;

    // 2. Update Status Banner
    statusBanner.className = `status-banner ${theme}`;
    if (theme === 'safe') {
      bannerIcon.textContent = '🛡️';
      bannerTitle.textContent = 'High Survival Probability';
      bannerDesc.textContent = `At ${pct}%, this passenger profile aligns strongly with successful historical evacuation protocols.`;
    } else if (theme === 'warning') {
      bannerIcon.textContent = '⚠️';
      bannerTitle.textContent = 'Moderate Survival Probability';
      bannerDesc.textContent = `At ${pct}%, survival depends on immediate access to upper boat stations and deck protocols.`;
    } else {
      bannerIcon.textContent = '🚨';
      bannerTitle.textContent = 'High Mortality Risk';
      bannerDesc.textContent = `At ${pct}%, demographic deferral or lower hull location severely impacted lifeboat access.`;
    }

    const now = new Date();
    resultsTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    // 3. Models Consensus
    const lrProb = pred.models.logistic_regression.probability;
    const rfProb = pred.models.random_forest.probability;

    lrScoreDisplay.textContent = `${lrProb}%`;
    lrProgressBar.style.width = `${lrProb}%`;

    rfScoreDisplay.textContent = `${rfProb}%`;
    rfProgressBar.style.width = `${rfProb}%`;

    // 4. Diagnostic Explanations
    factorsList.innerHTML = '';
    if (data.factors && data.factors.length > 0) {
      data.factors.forEach(f => {
        const item = document.createElement('div');
        item.className = `factor-item ${f.impact}`;
        const icon = f.impact === 'positive' ? '✓' : f.impact === 'negative' ? '✕' : 'ℹ';
        item.innerHTML = `
          <span class="factor-badge">${icon}</span>
          <div class="factor-content">
            <span class="factor-heading">${f.title} (${f.category})</span>
            <span class="factor-desc">${f.description}</span>
          </div>
        `;
        factorsList.appendChild(item);
      });
    }

    // 5. Boarding Pass Update
    passTicket.textContent = pass.ticket_code;
    passName.textContent = pass.name;
    passClass.textContent = pass.pclass_label;
    passPort.textContent = pass.embarked_city;
    passAgeGender.textContent = `${pass.age} yrs / ${pass.sex}`;
    passParty.textContent = `${pass.family_size} (${pass.is_alone ? 'Solo' : 'Group'})`;

    passSealRate.textContent = `${pct}%`;
    if (pred.survived) {
      passStatusSeal.className = 'pass-status-seal';
      passSealText.textContent = 'SURVIVOR';
    } else {
      passStatusSeal.className = 'pass-status-seal deceased';
      passSealText.textContent = 'PERISHED';
    }
  }

  // -------------------------------------------------------------------------
  // 4. Initialization
  // -------------------------------------------------------------------------
  updateAgeDisplay();
  updateFareDisplay();
  updateCompanionship();
  loadPresets();
  runPrediction();
});
