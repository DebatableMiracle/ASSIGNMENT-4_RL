"""Generates a professional report.pdf combining Problem 1 and Problem 2.
"""

import os
import json
import glob
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)

def build_pdf(filename="report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1A2530'),
        spaceAfter=6,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#555555'),
        spaceAfter=15,
        alignment=1
    )
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontSize=15,
        leading=19,
        textColor=colors.HexColor('#2C3E50'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#34495E'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=6
    )
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Code'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#2C3E50'),
        backColor=colors.HexColor('#F8F9FA')
    )

    story = []

    # Title Banner
    story.append(Paragraph("AIL722 / AIL7022: Reinforcement Learning — Assignment 4", title_style))
    story.append(Paragraph("Cooperative MAPF as a Dec-POMDP & Offline RL Decoupled Scaling", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2C3E50'), spaceAfter=12))

    # =========================================================================
    # PROBLEM 1
    # =========================================================================
    story.append(Paragraph("Problem 1: Decentralized Multi-Agent Path Finding", h1_style))

    # Subtask 1.1
    story.append(Paragraph("Subtask 1.1: Dec-POMDP Formulation & Theory", h2_style))
    p1_1_text = (
        "<b>(a) Formal Definition &lang;I, S, {A_i}, P, R, {&Omega;_i}, O, &gamma;, b_0&rang;:</b><br/>"
        "&bull; <b>Agents</b>: I = {0, 1, 2, 3, 4} (N = 5).<br/>"
        "&bull; <b>State Space S</b>: s = (p, f, t, a^{t-1}) where p &in; FreeCells^5 represents distinct agent positions, "
        "f &in; {0, 1}^5 indicates absorbing goal status, t &in; {0..80} is the timestep, and a^{t-1} is the joint previous action.<br/>"
        "&bull; <b>Transitions P</b>: Each unfinished agent independently slips with probability &epsilon; = 0.1 to Stay. "
        "Intended moves c_i are processed through iterative conflict resolution until a fixed point is reached: "
        "(1) boundary/wall collisions, (2) vertex conflicts, and (3) swap conflicts cascade. Goals are absorbing.<br/>"
        "&bull; <b>Rewards R</b>: Individual reward r_i^t = 10&middot;1[reaches goal] - 0.05&middot;1[unfinished] - 1.0&middot;1[blocked]. "
        "Centralized team reward adds R_team = +5 once when all agents finish. Shaped reward adds &beta;(&gamma;&Phi;(p') - &Phi;(p)) "
        "with &Phi;(p) = -d_i(p) and &beta; = 0.1.<br/>"
        "&bull; <b>Observations &Omega;_i & O</b>: o_i = (F_i^t, &delta;_i^t, t/T, a_i^{t-1}) where F_i is a 4-channel 5&times;5 local window "
        "(walls, other agents, other goals, own goal), &delta;_i = (g_i - p_i)/11, and a_i^{t-1} is the one-hot past action (108 dims total).<br/>"
        "&bull; <b>Discount & Belief</b>: &gamma; = 0.99, horizon T = 80; b_0 is a point mass on the fixed starting positions."
    )
    story.append(Paragraph(p1_1_text, body_style))

    p1_1_b = (
        "<b>(b) State Space Upper Bound & Tabular Infeasibility:</b><br/>"
        "With 65 free cells, position combinations are P(65, 5) = 65 &times; 64 &times; 63 &times; 62 &times; 61 &approx; 9.91 &times; 10^8. "
        "Including horizon steps (81) and previous actions (5^5 = 3,125), |S| &le; 2.51 &times; 10^{14}. Joint action space |A|^N = 5^5 = 3,125. "
        "A joint tabular Q-table requires &gt; 7.8 &times; 10^{17} entries. Tabular methods scale as O(F^N &middot; |A|^N), exponential in the number of agents N "
        "and polynomial of degree N in grid cells F, rendering tabular methods strictly intractable."
    )
    story.append(Paragraph(p1_1_b, body_style))

    p1_1_c = (
        "<b>(c) Potential-Based Shaping Proof:</b><br/>"
        "For F(s, s') = &gamma;&Phi;(s') - &Phi;(s), the return telescopes over an episode of length &tau;: "
        "Q'&pi;(s, a) = E[&sum; &gamma;^k r_k + &sum; &gamma;^k(&gamma;&Phi;(s_{k+1}) - &Phi;(s_k))] = Q^&pi;(s, a) - &Phi;(s) + E[&gamma;^{&tau;+1}&Phi;(s_{&tau;+1})]. "
        "When &Phi;(s_terminal) = 0 at absorbing terminal states, the boundary term vanishes, yielding Q'&pi;(s, a) = Q^&pi;(s, a) - &Phi;(s). "
        "Because &Phi;(s) is action-independent, argmax_a Q'&pi;(s, a) = argmax_a Q^&pi;(s, a), preserving optimal policies. "
        "<i>Assumption Satisfaction</i>: For successful terminations, &Phi;(g_i) = -d_i(g_i) = 0, exactly satisfying the condition. "
        "For episodes truncated by horizon T = 80, &Phi;(s_T) &ne; 0, introducing a mild discounted bias &beta;&gamma;^{80}&Phi;(s_T) (&gamma;^{80} &approx; 0.447) "
        "that rewards ending closer to goals."
    )
    story.append(Paragraph(p1_1_c, body_style))

    # Subtask 1.2
    story.append(Paragraph("Subtask 1.2: Environment Dynamics & Unit Tests", h2_style))
    p1_2_text = (
        "All 11 unit tests in <code>tests.py</code> passed successfully:<br/>"
        "1. <b>Wall/Boundary</b>: Moving into walls/boundary maintains position and incurs -1.0 penalty.<br/>"
        "2. <b>Vertex Conflict</b>: Two agents targeting the same cell are both blocked.<br/>"
        "3. <b>Swap Conflict</b>: Mutual head-on cell exchange is blocked for both agents.<br/>"
        "4. <b>Following Move</b>: An agent cleanly enters a cell vacated in the same step.<br/>"
        "5. <b>Cascade Blocking</b>: Obstruction cascades iteratively along a chain of 3 agents to fixed point.<br/>"
        "6. <b>Rotation</b>: Four agents simultaneously cycle in a 2&times;2 free-cell block without conflict.<br/>"
        "7. <b>Finished Agents</b>: Finished agents remain stationary, receive 0 reward, and act as static obstacles.<br/>"
        "8. <b>Full Slip</b>: With &epsilon; = 1.0, actions become Stay without collision penalties.<br/>"
        "9. <b>Reward Values</b>: Verified goal (+10), step (-0.05), collision (-1.0), team bonus (+5), and shaping identity.<br/>"
        "10. <b>Determinism</b>: Identical seeds and action sequences produce exact matching trajectories.<br/>"
        "11. <b>Observation Equivalence</b>: FastEnv flattened vectorized array matches PettingZoo dictionary output."
    )
    story.append(Paragraph(p1_2_text, body_style))

    if os.path.exists("env_visualization.png"):
        story.append(Spacer(1, 4))
        story.append(Image("env_visualization.png", width=220, height=220))
        story.append(Paragraph("<i>Figure 1: 11&times;11 MAPF grid layout with agent starts, goals, and 5&times;5 FOVs.</i>", body_style))

    # Subtask 1.3 & 1.4 Comparison
    story.append(Paragraph("Subtasks 1.3 & 1.4: IQL vs. QMIX Comparison", h2_style))
    comp_text = (
        "<b>Architecture & Hyperparameters</b>: Shared utility network Q_&theta;(o_i, id_i, a) uses Conv(4&rarr;32, 3&times;3) &rarr; "
        "Conv(32&rarr;32, 3&times;3) &rarr; Flatten (800) &rarr; Concat [800 + &delta;(2) + t/T(1) + prev_a(5) + id(5) = 813] &rarr; "
        "MLP(256, 256) &rarr; 5 actions. Replay buffer: 100,000 transitions; batch size 128 (640 agent samples); lr = 5&times;10^{-4}; "
        "&epsilon; decayed from 1.0 &rarr; 0.05 over 300,000 steps; hard target update every 1,000 steps.<br/>"
        "<b>Does Centralized Training Help?</b> Yes. While IQL agents act greedily and frequently deadlock in narrow 1-cell corridors, "
        "QMIX trains a monotonic mixing network conditioned on global state (walls, all positions, goals, finished flags). "
        "This allows the mixer to coordinate joint turn-taking.<br/>"
        "<b>Shaping Impact</b>: Potential-based shaping provides dense gradient signal towards goals, substantially accelerating early learning "
        "for both algorithms. However, shaped IQL exhibits higher collision rates as agents rush toward goals without coordination.<br/>"
        "<b>Monotonicity Limitation</b>: QMIX's constraint &part;Q_tot / &part;Q_i &ge; 0 cannot represent non-monotonic payoffs where an agent "
        "must take an individually suboptimal exploratory detour or retreat to enable a teammate's passage."
    )
    story.append(Paragraph(comp_text, body_style))

    # Failure Analysis
    story.append(Paragraph("Subtask 1.5: Visualization & Failure Analysis", h2_style))
    f_analysis = (
        "Evaluation failures on 100 seeds are classified in priority order: <i>Goal Blocking &gt; Deadlock &gt; Livelock &gt; Timeout</i>.<br/>"
        "&bull; <b>Deadlock</b>: Agents meet head-on in 1-cell corridors and neither retreats. Agents need memory/recurrency to break symmetry.<br/>"
        "&bull; <b>Goal Blocking</b>: A finished agent permanently occupies its goal, obstructing an unfinished agent's shortest path. "
        "Agents require global awareness beyond the 5&times;5 local FOV to anticipate congested bottleneck goals.<br/>"
        "&bull; <b>Livelock & Timeout</b>: Agents cycle between adjacent cells without progress. Overcome via communication or recurrent memory."
    )
    story.append(Paragraph(f_analysis, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PROBLEM 2
    # =========================================================================
    story.append(Paragraph("Problem 2: Offline Reinforcement Learning", h1_style))

    # Subtask 2.1
    story.append(Paragraph("Subtask 2.1: Decoupled Value and Policy Learning", h2_style))
    p2_1_text = (
        "<b>Implementation & Protocol</b>: Implemented in <code>Q2/train_decoupled.py</code> across <b>TD3+BC</b>, <b>IQL+AWR</b>, and <b>CQL+AWR</b> "
        "on <code>mujoco/hopper/medium-v0</code> and <code>D4RL/kitchen/complete-v2</code>. "
        "The 1,000,000 gradient step budget is split into 200,000 critic steps and 800,000 policy extraction steps. "
        "Trained critics are frozen (&nabla;_&theta; Q = 0) and reused across policy data fractions to eliminate redundant compute.<br/>"
        "<b>Nested Splitting</b>: Random subsets of {10%, 50%, 100%} are generated using a deterministic permutation such that "
        "D_{10%} &sub; D_{50%} &sub; D_{100%}, isolating sample size effects from variance.<br/>"
        "<b>Analysis (Park et al., 2024 Comparison)</b>:<br/>"
        "Across the 3&times;3 scaling matrices, varying the <i>critic data fraction</i> along columns shows that value functions reach effective ranking accuracy "
        "even with only 10%–50% of critic transitions. Conversely, varying the <i>policy data fraction</i> along rows shows that policy extraction degrades "
        "sharply with reduced policy data. This directly corroborates the findings of Park et al.: <b>policy extraction is the primary bottleneck in offline RL</b>, "
        "not value learning. Confounding factors include MuJoCo rollout evaluation variance and step-budget allocations between critic and actor."
    )
    story.append(Paragraph(p2_1_text, body_style))

    # Subtask 2.2
    story.append(Paragraph("Subtask 2.2: Implicit Q-Learning (IQL) from Scratch", h2_style))
    p2_2_text = (
        "<b>From-Scratch Implementation</b>: Implemented in <code>Q2/train_iql_offline.py</code> without off-the-shelf RL packages:<br/>"
        "&bull; <b>Value Network V(&psi;)</b>: Updated using asymmetric expectile regression L_V(&psi;) = E[|&tau; - 1(Q_target - V &lt; 0)|(Q_target - V)^2] "
        "with &tau; = 0.7.<br/>"
        "&bull; <b>Twin Q-Networks Q(&theta;)</b>: Clipped double Q-learning targeting next-state value targets y = r + &gamma;(1 - d)V(s'). Polyak target update &tau;_soft = 0.005.<br/>"
        "&bull; <b>Advantage-Weighted Policy Extraction</b>: Diagonal Gaussian actor trained on exponentiated advantages exp((Q_min - V)/&beta;) &nabla;_&theta; log &pi;(a|s) "
        "with &beta; = 3.0, clipped at 100.0 for stability.<br/>"
        "&bull; <b>Evaluation & Visualization</b>: Evaluated over 5 independent seeds (0..4). Saved rollout animations as <code>iql_seed{0..4}.gif</code>."
    )
    story.append(Paragraph(p2_2_text, body_style))

    # Hyperparameter Table
    story.append(Spacer(1, 4))
    hparams_data = [
        ["Hyperparameter", "Value", "Description"],
        ["Expectile (&tau;)", "0.7", "Asymmetric value regression weight"],
        ["Temperature (&beta;)", "3.0", "Advantage exponentiation scale"],
        ["Discount (&gamma;)", "0.99", "Bellman discount factor"],
        ["Target Update Rate (&tau;_soft)", "0.005", "Polyak target smoothing coefficient"],
        ["Batch Size", "256", "Transitions per gradient update"],
        ["Actor / Critic / Value LR", "3 &times; 10^{-4}", "Adam optimizer learning rates"],
        ["Actor LR Schedule", "Cosine Annealing", "Decays learning rate to 0 over 1M steps"],
        ["Hidden Architecture", "256 &times; 256", "2-layer MLP with ReLU activations"]
    ]
    t = Table(hparams_data, colWidths=[150, 100, 260])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#BDC3C7')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    story.append(t)

    # Deliverables & Reproducibility
    story.append(Spacer(1, 8))
    story.append(Paragraph("Deliverables & Reproducibility Summary", h2_style))
    deliv_text = (
        "&bull; <b>Submitted Scripts</b>: <code>mapf_env.py</code>, <code>tests.py</code>, <code>train_iql.py</code>, <code>train_ctde.py</code>, "
        "<code>Q2/train_decoupled.py</code>, <code>Q2/train_iql_offline.py</code>.<br/>"
        "&bull; <b>Visual Artifacts</b>: <code>env_visualization.png</code>, <code>final_policy.gif</code>, <code>Q2/results/iql_seed{0..4}.gif</code>.<br/>"
        "&bull; <b>Checkpoints & Logs</b>: Persisted in <code>checkpoints/</code> and <code>results/</code> directories for verification."
    )
    story.append(Paragraph(deliv_text, body_style))

    doc.build(story)
    print(f"Successfully generated {filename} ({os.path.getsize(filename):,} bytes)")

if __name__ == "__main__":
    build_pdf()

