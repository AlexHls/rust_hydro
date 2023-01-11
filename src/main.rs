extern crate npy;

use cpu_time::ProcessTime;
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

//Global parameter
const NX: usize = 128;
const NY: usize = 64;
const N_GHOST: usize = 2;

const GRIDSIZE: usize = NX + 2 * N_GHOST;
const NVARS: usize = 5;

const I_RHO: usize = 0;
const I_P: usize = 1;
const I_U: usize = 2;
const I_V: usize = 3;
const I_PS1: usize = 4;

const CFL: f64 = 0.4;
const GAMMA_AD: f64 = 1.4;

const X1: f64 = 0.0;
const Y1: f64 = -0.5;
const DL: f64 = 1.0 / NY as f64;
const MACH_X: f64 = 0.2;
const TMAX: f64 = 0.8 / MACH_X;
const PDIFF: f64 = 1.0;

const TER_FRE: i32 = 300;
const OUT_FRE: i32 = 30000000;

const INV_DL: f64 = 1.0 / DL;
const GAMMAM1: f64 = GAMMA_AD - 1.0;
const INV_GAMMAM1: f64 = 1.0 / (GAMMA_AD - 1.0);

fn apply_bcs(data_cc: &mut [[[f64; GRIDSIZE]; GRIDSIZE]; NVARS]) {
    // Apply boundary conditions
    for j in 0..NY {
        for iv in 0..NVARS {
            data_cc[iv][GRIDSIZE - 1][j] = data_cc[iv][NX - 1][j];
            data_cc[iv][0][j] = data_cc[iv][NX][j];
            data_cc[iv][NX + 1][j] = data_cc[iv][1][j];
            data_cc[iv][NX + 2][j] = data_cc[iv][2][j];
        }
    }
    for i in 0..NX {
        for iv in 0..NVARS {
            data_cc[iv][i][GRIDSIZE - 1] = data_cc[iv][i][NY - 1];
            data_cc[iv][i][0] = data_cc[iv][i][NY];
            data_cc[iv][i][NY + 1] = data_cc[iv][i][1];
            data_cc[iv][i][NY + 2] = data_cc[iv][i][2];
        }
    }
}

fn dump_output(data: &[[[f64; GRIDSIZE]; GRIDSIZE]; NVARS], nout: i32) -> std::io::Result<()> {
    let rho_pathname = format!("data/rho_{:?}.dat", nout);
    let rho_path = Path::new(&rho_pathname);
    let rho_display = rho_path.display();
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut rho_file = match File::create(&rho_path) {
        Err(why) => panic!("couldn't create {}: {}", rho_display, why),
        Ok(rho_file) => rho_file,
    };

    let u_pathname = format!("data/u_{:?}.dat", nout);
    let u_path = Path::new(&u_pathname);
    let u_display = u_path.display();
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut u_file = match File::create(&u_path) {
        Err(why) => panic!("couldn't create {}: {}", u_display, why),
        Ok(u_file) => u_file,
    };

    let v_pathname = format!("data/v_{:?}.dat", nout);
    let v_path = Path::new(&v_pathname);
    let v_display = v_path.display();
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut v_file = match File::create(&v_path) {
        Err(why) => panic!("couldn't create {}: {}", v_display, why),
        Ok(v_file) => v_file,
    };

    let p_pathname = format!("data/p_{:?}.dat", nout);
    let p_path = Path::new(&p_pathname);
    let p_display = p_path.display();
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut p_file = match File::create(&p_path) {
        Err(why) => panic!("couldn't create {}: {}", p_display, why),
        Ok(p_file) => p_file,
    };

    let ps1_pathname = format!("data/ps1_{:?}.dat", nout);
    let ps1_path = Path::new(&ps1_pathname);
    let ps1_display = ps1_path.display();
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut ps1_file = match File::create(&ps1_path) {
        Err(why) => panic!("couldn't create {}: {}", ps1_display, why),
        Ok(ps1_file) => ps1_file,
    };

    // Write to `file`, returns `io::Result<()>`
    for j in 0..NY {
        for i in 0..NX {
            rho_file.write_all(&data[I_RHO][i][j].to_be_bytes());
        }
        for i in 0..NX {
            u_file.write_all(&data[I_U][i][j].to_be_bytes());
        }
        for i in 0..NX {
            v_file.write_all(&data[I_V][i][j].to_be_bytes());
        }
        for i in 0..NX {
            p_file.write_all(&data[I_P][i][j].to_be_bytes());
        }
        for i in 0..NX {
            ps1_file.write_all(&data[I_PS1][i][j].to_be_bytes());
        }
    }
    Ok(())
}

fn riemann(
    nf: usize,
    dir: i32,
    q_l: &mut [[f64; NX + 1]; NVARS],
    q_r: &mut [[f64; NX + 1]; NVARS],
    flux: &mut [[f64; NX + 1]; NVARS],
) {
    let mut sgn: f64;
    let mut rho_l: f64;
    let mut u_l: f64;
    let mut v_l: f64;
    let mut p_l: f64;
    let mut e_l: f64;
    let mut c_l: f64;
    let mut rho_r: f64;
    let mut u_r: f64;
    let mut v_r: f64;
    let mut p_r: f64;
    let mut e_r: f64;
    let mut c_r: f64;

    let mut s_l: f64;
    let mut s_r: f64;
    let mut inv_rho_l: f64;
    let mut inv_rho_r: f64;
    let mut dsu_l: f64;
    let mut dsu_r: f64;
    let mut ustar: f64;
    let mut rhostar: f64;
    let mut rhostar_l: f64;
    let mut rhostar_r: f64;
    let mut rhobar: f64;
    let mut cbar: f64;
    let mut mach_l: f64;
    let mut mach_r: f64;
    let mut phi: f64;
    let mut chi: f64;
    let mut pstar: f64;
    let mut estar_l: f64;
    let mut estar_r: f64;

    for idxf in 0..nf {
        if dir == 1 {
            u_l = q_l[I_U][idxf];
            v_l = q_l[I_V][idxf];

            u_r = q_r[I_U][idxf];
            v_r = q_r[I_V][idxf];
        } else if dir == 2 {
            u_l = q_l[I_V][idxf];
            v_l = -q_l[I_U][idxf];

            u_r = q_r[I_V][idxf];
            v_r = -q_r[I_U][idxf];
        } else {
            u_l = 0.0;
            v_l = 0.0;

            u_r = 0.0;
            v_r = 0.0;
        }

        rho_l = q_l[I_RHO][idxf];
        p_l = q_l[I_P][idxf];
        e_l = p_l * INV_GAMMAM1 + 0.5 * rho_l * (u_l.powi(2) + v_l.powi(2));

        rho_r = q_r[I_RHO][idxf];
        p_r = q_r[I_P][idxf];
        e_r = p_r * INV_GAMMAM1 + 0.5 * rho_r * (u_r.powi(2) + v_r.powi(2));

        inv_rho_l = 1.0 / rho_l;
        inv_rho_r = 1.0 / rho_r;

        c_l = (GAMMA_AD * p_l * inv_rho_l).sqrt();
        c_r = (GAMMA_AD * p_r * inv_rho_r).sqrt();

        s_l = u_l.min(u_r) - c_l.max(c_r);
        s_r = u_l.max(u_r) + c_l.max(c_r);

        dsu_l = s_l - u_l;
        dsu_r = s_r - u_r;

        ustar = (p_r - p_l + rho_l * u_l * dsu_l - rho_r * u_r * dsu_r)
            / (rho_l * dsu_l - rho_r * dsu_r);

        mach_l = (u_l.powi(2) + v_l.powi(2)).sqrt() / c_l;
        mach_r = (u_r.powi(2) + v_r.powi(2)).sqrt() / c_r;

        chi = mach_l.max(mach_r);

        phi = chi * (2.0 - chi);

        rhobar = 0.5 * (rho_l + rho_r);
        cbar = 0.5 * (c_l + c_r);

        pstar = 0.5 * (p_l + p_r) - PDIFF * 0.5 * phi * rhobar * cbar * (u_r - u_l);

        sgn = 1.0 * ustar.signum();

        rhostar_l = rho_l * (dsu_l / (s_l - ustar));
        estar_l = rhostar_l * (e_l * inv_rho_l + (ustar - u_l) * (ustar + p_l * inv_rho_l / dsu_l));

        rhostar_r = rho_r * (dsu_r / (s_r - ustar));
        estar_r = rhostar_r * (e_r * inv_rho_r + (ustar - u_r) * (ustar + p_r * inv_rho_r / dsu_r));

        rhostar = 0.5 * (1.0 + sgn) * rhostar_l + 0.5 * (1.0 - sgn) * rhostar_r;

        flux[I_RHO][idxf] = rhostar * ustar;
        flux[I_P][idxf] =
            (0.5 * (1.0 + sgn) * estar_l + 0.5 * (1.0 - sgn) * estar_r + pstar) * ustar;
        flux[I_PS1][idxf] = rhostar
            * ustar
            * (0.5 * (1.0 + sgn) * q_l[I_PS1][idxf] + 0.5 * (1.0 - sgn) * q_r[I_PS1][idxf]);

        if dir == 1 {
            flux[I_U][idxf] = rhostar * ustar.powi(2) + pstar;
            flux[I_V][idxf] = rhostar * ustar * 0.5 * ((1.0 + sgn) * v_l + (1.0 - sgn) * v_r);
        } else if dir == 2 {
            flux[I_U][idxf] = -(rhostar * ustar * 0.5 * ((1.0 + sgn) * v_l + (1.0 - sgn) * v_r));
            flux[I_V][idxf] = rhostar * ustar.powi(2) + pstar;
        }
    }
}

fn main() {
    // Variables
    let mut t: f64 = 0.0;
    let mut xi: f64;
    let mut yj: f64;
    let mut rho: f64;
    let mut u: f64;
    let mut v: f64;
    let mut rhou: f64;
    let mut rhov: f64;
    let mut inv_rho: f64;
    let mut m_l: f64;
    let mut m_r: f64;
    let mut sigma: f64;
    let mut qm2: f64;
    let mut qm1: f64;
    let mut q: f64;
    let mut qp1: f64;
    let mut eta: f64;

    let mut prim: [[[f64; GRIDSIZE]; GRIDSIZE]; NVARS] = [[[0.0; GRIDSIZE]; GRIDSIZE]; NVARS];
    let mut cons0: [[[f64; NY]; NX]; NVARS] = [[[0.0; NY]; NX]; NVARS];
    let mut cons: [[[f64; NY]; NX]; NVARS] = [[[0.0; NY]; NX]; NVARS];

    let mut q_l: [[f64; NX + 1]; NVARS] = [[0.0; NX + 1]; NVARS];
    let mut q_r: [[f64; NX + 1]; NVARS] = [[0.0; NX + 1]; NVARS];
    let mut flux: [[f64; NX + 1]; NVARS] = [[0.0; NX + 1]; NVARS];

    let mut dir: i32;
    let mut nsteps: i32 = 0;
    let mut nout: i32 = 0;

    // CC
    for j in 0..NY {
        for i in 0..NX {
            prim[I_RHO][i][j] = GAMMA_AD;
            prim[I_P][i][j] = 1.0;

            xi = X1 + DL * (i as f64 - 0.5);
            yj = Y1 + DL * (i as f64 - 0.5);

            if yj > (-0.25 - 1.0 / 32.0) && yj < (-0.25 + 1.0 / 32.0) {
                eta = 0.5 * (1.0 + (16.0 * PI * (yj + 0.25)).sin());
            } else if yj >= (-0.25 + 1.0 / 32.0) && yj <= (0.25 - 1.0 / 32.0) {
                eta = 1.0;
            } else if yj > (0.25 - 1.0 / 32.0) && yj < (0.25 + 1.0 / 32.0) {
                eta = 0.5 * (1.0 + (-16.0 * PI * (yj + 0.25)).sin());
            } else {
                eta = 0.0;
            }

            prim[I_U][i][j] = MACH_X * (1.0 - 2.0 * eta);
            prim[I_V][i][j] = 0.1 * MACH_X * (2.0 * PI * xi).sin();

            prim[I_PS1][i][j] = eta;
        }
    }

    let dt: f64 = 0.5 * CFL * DL / (1.0 + MACH_X);

    println!("t: {t}, dt: {dt}");

    let inv_dl_dthalf: f64 = INV_DL * dt * 0.5;
    let inv_dl_dt: f64 = INV_DL * dt;

    apply_bcs(&mut prim);

    let mut t1 = ProcessTime::try_now().expect("Getting process time failed");

    dump_output(&prim, nout);

    let mut t2 = t1.elapsed();

    println!("OUTPUT WCT [s]: {:?}", t2);

    // Main loop

    t1 = ProcessTime::try_now().expect("Getting process time failed");

    while t < TMAX {
        if nsteps % TER_FRE == 0 {
            println!("t: {}, t/tmax: {}", t, t / TMAX);
        }

        if nsteps % OUT_FRE == 0 {
            dump_output(&prim, nout);
            nout += 1;
        }

        for j in 0..NY {
            for i in 0..NX {
                rho = prim[I_RHO][i][j];
                u = prim[I_U][i][j];
                v = prim[I_V][i][j];

                cons0[I_RHO][i][j] = rho;
                cons0[I_U][i][j] = rho * u;
                cons0[I_V][i][j] = rho * v;

                cons0[I_P][i][j] =
                    prim[I_P][i][j] * INV_GAMMAM1 + 0.5 * rho * (u.powi(2) + v.powi(2));
                cons0[I_PS1][i][j] = prim[I_PS1][i][j] * rho;
            }
        }

        // Predictor

        apply_bcs(&mut prim);

        dir = 1;

        for j in 0..NY {
            for i in 0..=NX {
                for iv in 0..NVARS {
                    q_l[iv][i] = prim[iv][(i + GRIDSIZE - 1) % GRIDSIZE][j];
                    q_r[iv][i] = prim[iv][i][j];
                }
            }
            riemann(NX + 1, dir, &mut q_l, &mut q_r, &mut flux);

            for i in 0..NX {
                for iv in 0..NVARS {
                    cons[iv][i][j] = cons0[iv][i][j]
                        - (flux[iv][(i + 1) % GRIDSIZE] - flux[iv][i]) * inv_dl_dthalf;
                }
            }
        }

        dir = 2;

        for i in 1..NX {
            for j in 0..=NY {
                for iv in 0..NVARS {
                    q_l[iv][j] = prim[iv][i][(j + GRIDSIZE - 1) % GRIDSIZE];
                    q_r[iv][j] = prim[iv][i][j];
                }
            }
            riemann(NY + 1, dir, &mut q_l, &mut q_r, &mut flux);

            for j in 0..NY {
                for iv in 0..NVARS {
                    cons[iv][i][j] = cons[iv][i][j]
                        - (flux[iv][(j + 1) % GRIDSIZE] - flux[iv][j]) * inv_dl_dthalf;
                }
            }
        }

        for j in 0..NY {
            for i in 0..NX {
                rho = cons[I_RHO][i][j];
                inv_rho = 1.0 / rho;
                rhou = cons[I_U][i][j];
                rhov = cons[I_V][i][j];

                prim[I_RHO][i][j] = rho;
                prim[I_U][i][j] = rhou * inv_rho;
                prim[I_V][i][j] = rhov * inv_rho;
                prim[I_P][i][j] =
                    GAMMAM1 * (cons[I_P][i][j] - 0.5 * inv_rho * (rhou.powi(2) + rhov.powi(2)));
                prim[I_PS1][i][j] = cons[I_PS1][i][j] * inv_rho;
            }
        }

        // Corrector

        dir = 1;

        for j in 0..NY {
            for i in 0..NX {
                for iv in 0..NVARS {
                    qm2 = prim[iv][(i + GRIDSIZE - 2) % GRIDSIZE][j];
                    qm1 = prim[iv][(i + GRIDSIZE - 1) % GRIDSIZE][j];
                    q = prim[iv][i][j];
                    qp1 = prim[iv][(i + 1) % GRIDSIZE][j];

                    // left
                    m_l = qm1 - qm2;
                    m_r = q - qm1;

                    sigma = 0.5 * (m_l + m_r);

                    q_l[iv][i] = qm1 + 0.5 * sigma;

                    // right
                    m_l = m_r;
                    m_r = qp1 - q;

                    sigma = 0.5 * (m_l + m_r);

                    q_r[iv][i] = q - 0.5 * sigma;
                }
            }

            riemann(NX + 1, dir, &mut q_l, &mut q_r, &mut flux);

            for i in 0..NX {
                for iv in 0..NVARS {
                    cons[iv][i][j] =
                        cons0[iv][i][j] - (flux[iv][(i + 1) % GRIDSIZE] - flux[iv][i]) * inv_dl_dt;
                }
            }
        }

        dir = 2;

        for i in 0..NX {
            for j in 0..=NY {
                for iv in 0..NVARS {
                    qm2 = prim[iv][i][(j + GRIDSIZE - 2) % GRIDSIZE];
                    qm1 = prim[iv][i][(j + GRIDSIZE - 1) % GRIDSIZE];
                    q = prim[iv][i][j];
                    qp1 = prim[iv][i][(j + 1) % GRIDSIZE];

                    // left
                    m_l = qm1 - qm2;
                    m_r = q - qm1;

                    sigma = 0.5 * (m_l + m_r);

                    q_l[iv][j] = qm1 + 0.5 * sigma;

                    // right
                    m_l = m_r;
                    m_r = qp1 - q;

                    sigma = 0.5 * (m_l + m_r);

                    q_r[iv][j] = q - 0.5 * sigma;
                }
            }

            riemann(NY + 1, dir, &mut q_l, &mut q_r, &mut flux);

            for j in 0..NY {
                for iv in 0..NVARS {
                    cons[iv][i][j] =
                        cons0[iv][i][j] - (flux[iv][(j + 1) % GRIDSIZE] - flux[iv][j]) * inv_dl_dt;
                }
            }
        }

        for j in 0..NY {
            for i in 0..NX {
                rho = cons[I_RHO][i][j];
                inv_rho = 1.0 / rho;
                rhou = cons[I_U][i][j];
                rhov = cons[I_V][i][j];

                prim[I_RHO][i][j] = rho;
                prim[I_U][i][j] = rhou * inv_rho;
                prim[I_V][i][j] = rhov * inv_rho;
                prim[I_P][i][j] =
                    GAMMAM1 * (cons[I_P][i][j] - 0.5 * inv_rho * (rhou.powi(2) + rhov.powi(2)));
                prim[I_PS1][i][j] = cons[I_PS1][i][j] * inv_rho;
            }
        }

        t += dt;
        nsteps += 1;
    }

    dump_output(&mut prim, nout);

    t2 = t1.elapsed();

    println!("WCT [s]: {:?}", t2);
    //println!("WCT/cell/step [mus]: {:?}", t2 as f64 / (NX as f64* NY as f64)/nsteps as f64 * 1e6);
}
