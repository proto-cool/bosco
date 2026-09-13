/* bosco LIF kernel: deterministic, single-threaded, CSR by presynaptic neuron.
 *
 * Model (Shiu et al. 2024, Brian2 'linear' method, alpha-less exponential synapse):
 *   dv/dt = (v0 - v + g) / tau_m     (unless refractory)
 *   dg/dt = -g / tau_s               (unless refractory)
 *   spike if v > v_th; reset v = v_rst, g = 0; refractory t_rfc
 *   on presynaptic spike, after delay t_dly:  g_post += w   (w in mV, signed)
 *
 * Step order mirrors Brian2's default schedule: state update, threshold,
 * synapses (delayed spike delivery + Poisson inputs), reset.
 *
 * All floating-point accumulation happens in a fixed order, so a run is
 * bit-identical for identical (weights, inputs, seed) on the same binary.
 */
#ifndef BOSCO_LIF_H
#define BOSCO_LIF_H
#include <stdint.h>

typedef struct {
    double dt_ms;    /* 0.1 */
    double tau_m;    /* 20 ms */
    double tau_s;    /* 5 ms */
    double v0;       /* -52 mV */
    double v_rst;    /* -52 mV */
    double v_th;     /* -45 mV */
    double t_rfc;    /* 2.2 ms */
    double t_dly;    /* 1.8 ms */
} lif_params;

typedef struct lif_net lif_net;

/* CSR by presynaptic neuron. indptr length n+1, indices/w length nnz.
 * Weights are copied; the CSR structure is copied too. */
lif_net *lif_create(int32_t n, const int64_t *indptr, const int32_t *indices,
                    const double *w, const lif_params *p);
void lif_free(lif_net *net);

/* Reset all dynamic state (v, g, refractory, delay ring, counters) and seed RNG. */
void lif_reset(lif_net *net, uint64_t seed);

/* Overwrite weights at given edge indices (positions in the CSR data array). */
void lif_set_weights(lif_net *net, const int64_t *edge_idx, const double *w, int64_t m);
/* Read all weights out (length nnz). */
void lif_get_weights(const lif_net *net, double *w_out);
int64_t lif_nnz(const lif_net *net);

/* Poisson (Bernoulli-per-step) inputs: neuron idx[i] receives, with
 * probability rate_hz[i]*dt each step, an instantaneous jump v += weight_mv[i].
 * Neurons with an input have their refractory period set to zero, as in Shiu.
 * Replaces any previous input set. */
void lif_set_inputs(lif_net *net, int32_t m, const int32_t *idx,
                    const double *rate_hz, const double *weight_mv);

/* Run n_steps. Spikes are appended to (t_out, id_out) up to max_spikes; the
 * return value is the number of spikes that occurred (may exceed max_spikes,
 * in which case the surplus were not recorded but still counted). */
int64_t lif_run(lif_net *net, int64_t n_steps, int32_t *t_out, int32_t *id_out,
                int64_t max_spikes);

/* Per-neuron spike counts since last reset (length n). */
void lif_spike_counts(const lif_net *net, int64_t *counts);
/* Copy membrane potentials (length n). */
void lif_get_v(const lif_net *net, double *v_out);
int64_t lif_step(const lif_net *net);

#endif
