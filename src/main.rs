// Desc: Main entry point for the program
use rand::Rng;

const PARTICLES: usize = 30; // number of particles
const ITERATIONS: usize = 100; // number of iterations
const PARAMETERS: usize = 10; // number of dimensions
const C1: f64 = 1.3; // acceleration constant c1 (cognitive component)
const C2: f64 = 1.1; // acceleration constant c2 (social component)
const W: f64 = 0.9; // inertia weight
const LOWER_BOUND: f64 = -1.0; // lower bound of the search space (Wmin)
const UPPER_BOUND: f64 = 1.0; // upper bound of the search space (Wmax)
const PENALTY_FACTOR: f64 = 10000.0; // penalty for particles out-of-the-bounds

struct Particle {
    x: Vec<f64>,
    vx: Vec<f64>,
    pbest_x: Vec<f64>,
    pbest: f64,
}

struct Swarm {
    particles: Vec<Particle>,
    gbest_x: Vec<f64>,
    gbest: f64,
}

fn run_pso(swarm: &mut Swarm) {
    let mut rng = rand::thread_rng();

    for it in 0..ITERATIONS {
        for i in 0..PARTICLES {
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();

            let p = &mut swarm.particles[i];

            let mut penalty: bool = false;

            // update velocity and position
            for pos in 0..PARAMETERS {
                p.vx[pos] = W * p.vx[pos]
                    + C1 * r1 * (p.pbest_x[pos] - p.x[pos])
                    + C2 * r2 * (swarm.gbest_x[pos] - p.x[pos]);
                p.x[pos] += p.vx[pos];
                if p.x[pos] < LOWER_BOUND {
                    p.x[pos] = LOWER_BOUND;
                    p.vx[pos] = 0.0;
                    penalty = true;
                } else if p.x[pos] > UPPER_BOUND {
                    p.x[pos] = UPPER_BOUND;
                    p.vx[pos] = 0.0;
                    penalty = true;
                }
            }

            // check if new position is better than previous position
            let mut fitness = rastrigin(&p.x);

            if penalty {
                fitness += PENALTY_FACTOR;
            }

            if fitness < p.pbest {
                p.pbest_x = p.x.clone();
                p.pbest = fitness;
            }

            // check if new position is better than global best
            if fitness < swarm.gbest {
                swarm.gbest_x = p.x.clone();
                swarm.gbest = fitness;
            }
        }

        if it % 100 == 0 {
            println!("Iteration: {}, gbest: {}", it, swarm.gbest);
        }

        save_fitness_to_csv(it, swarm.gbest);
    }

    println!("Best solution found at: fitness = {}", swarm.gbest);
    for i in 0..PARAMETERS {
        println!("x{}: {}", i + 1, swarm.gbest_x[i]);
    }
}

fn rastrigin(x: &Vec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i].powi(2) - 10.0 * (2.0 * std::f64::consts::PI * x[i]).cos();
    }

    10.0 * x.len() as f64 + sum
}

fn main() {
    println!("Rastrigin using Particle Swarm Optimization");
    println!("===========================================\n");

    print_params();

    let mut swarm = Swarm {
        particles: Vec::new(),
        gbest_x: vec![],
        gbest: 0.0,
    };

    init_swarm(&mut swarm);
    run_pso(&mut swarm);
}

fn init_swarm(s: &mut Swarm) {
    let mut rng = rand::thread_rng();

    // Initialize particles
    for _ in 0..PARTICLES {
        let mut params = Vec::new();
        let mut velocity = vec![];
        for _ in 0..PARAMETERS {
            let x = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
            let vx = rng.gen_range(LOWER_BOUND..UPPER_BOUND);
            params.push(x);
            velocity.push(vx);
        }

        let p = Particle {
            x: params.clone(),
            vx: velocity,
            pbest_x: params.clone(),
            pbest: rastrigin(&params),
        };

        s.particles.push(p);
    }

    // Set the global best to the first particle
    s.gbest_x = s.particles[0].x.clone();
    s.gbest = s.particles[0].pbest;
}

fn print_params() {
    println!("Parameters:");
    println!("  Number of particles: {}", PARTICLES);
    println!("  Number of iterations: {}", ITERATIONS);
    println!("  Inertia weight: {}", W);
    println!("  Cognitive weight: {}", C1);
    println!("  Social weight: {}", C2);
    println!(
        "  Lower and Upper bounds: [{}, {}]",
        LOWER_BOUND, UPPER_BOUND
    );
    println!("");
}

fn save_fitness_to_csv(iteration: usize, fitness: f64) {
    let filename = format!("test_9.csv");

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&filename)
        .expect("Error opening CSV file");

    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);

    writer
        .serialize((iteration, fitness))
        .expect("Error writing to CSV file");
}
