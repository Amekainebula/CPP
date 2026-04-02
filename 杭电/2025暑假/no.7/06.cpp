// Code from Whalica
#include <bits/stdc++.h>

using i64 = long long;
using u64 = unsigned long long;

void solve()
{
    i64 k, n, a, b, c, d;
    std::cin >> k >> n >> a >> b >> c >> d;

    auto le = [&](i64 w, i64 x, i64 y, i64 z) -> bool
    {
        return w * z < y * x;
    };

    auto leq = [&](i64 w, i64 x, i64 y, i64 z) -> bool
    {
        return w * z <= y * x;
    };

    i64 p = a, q = b, ansp = 0, ansq = 1;
    while (leq(p, q, c, d))
    {
        i64 gg = std::gcd(a, b);
        a /= gg, b /= gg;
        //		std::cout << "p / q : " << p << "/" << q << "\n";
        i64 x = p * n, w = k * q - (k * q) % x; // w / x
        i64 g1 = std::gcd(x, w);
        x /= g1, w /= g1;

        //		std::cout << "val : " << w << "/" << x << "\n";

        i64 z = w * n, y = k * x; // y / z
        i64 g2 = std::gcd(y, z);
        y /= g2, z /= g2;

        if (le(c, d, y, z))
        {
            y = c, z = d;
        }

        //		std::cout << "right bound : " << y << "/" << z << "\n";

        w += x;
        w *= y, x *= z;
        i64 g3 = std::gcd(x, w);
        x /= g3, w /= g3;
        if (le(ansp, ansq, w, x))
        {
            ansp = w, ansq = x;
        }
        i64 e = std::lcm(z, d);
        e /= z;
        p = y * e + 1, q = z * e;
    }

    std::cout << ansp << "/" << ansq << "\n";
    //std::cout<<std::endl;
}

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t = 1;
    std::cin >> t;

    while (t--)
    {
        solve();
    }

    return 0;
}
