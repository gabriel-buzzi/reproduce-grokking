# %%
import jax
import jax.numpy as jnp
import functools as ft
import itertools as it
from matplotlib import pyplot as plt
from tqdm import tqdm

# %%
def model(params, x):

    h = x @ params["W_1"] + params["b_1"]

    h = jax.nn.relu(h)

    h = h @ params["W_2"] + params["b_2"]

    return h

#%%
root_key = jax.random.key(43)
key = map(ft.partial(jax.random.fold_in, root_key), it.count())

params = {
    "W_1": jax.random.normal(next(key), (1, 10)),
    "b_1": jax.random.normal(next(key), (10, )),
    "W_2": jax.random.normal(next(key), (10, 1)),
    "b_2": jax.random.normal(next(key), (1, ))
}

print(params)
# %%
x = jnp.linspace(0, 3*2*jnp.pi)[:, jnp.newaxis]
y = jnp.sin(x)

plt.plot(x, y)

# %%
model(params, x)

# %%
def loss_fn(params, x, y):
    
    y_hat = model(params, x)

    return jnp.mean((y - y_hat)**2)

# %%
n_steps = 1000
lr = 1e-2

for i in tqdm(range(n_steps)):
    loss, grad = jax.value_and_grad(loss_fn, argnums=0)(params, x, y)

    params = jax.tree.map(lambda w, d: w - lr*d, params, grad)

    y_hat = model(params, x)

    plt.plot(y, label="ref")
    plt.plot(y_hat, label="pred")
    plt.savefig("fig.png")
    plt.close()
# %%