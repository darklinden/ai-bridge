# The client's model is always overridden by the profile's

Every local entry ignores the `model` the client sent and substitutes the profile's `model` before the request is converted or forwarded. A client cannot address a model the profile did not configure.

This is deliberate. The proxy is a bridge to *one* configured upstream, and honouring a client-supplied name would either fail upstream or — worse — silently reach a model the operator never chose, an aliasing problem no client can see from its side. The cost is that a client's model choice is a no-op, so the profile is the only place a model name ever enters the system.
