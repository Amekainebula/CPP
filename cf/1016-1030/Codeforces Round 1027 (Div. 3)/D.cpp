#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define vc vector
#define vi vector<int>
#define vvi vector<vi>
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define all1(x) x.begin() + 1, x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
// #define endl endl << flush
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
const int MOD = 1e9 + 7;
const int mod = 998244353;
const int N = 2e5 + 5;
void Murasame()
{
    int n;
    cin >> n;
    vc<pii> v(n + 1), x(n + 1), y(n + 1);
    ff(i, 1, n)
    {
        cin >> v[i].fi >> v[i].se;
        x[i] = {v[i].fi, i};
        y[i] = {v[i].se, i};
    }
    auto cmp = [&](pii a, pii b)
    { return a.fi < b.fi; };
    sort(all1(x), cmp);
    sort(all1(y), cmp);
    int min1 = x[1].fi, max1 = x[n].fi;
    int min2 = y[1].fi, max2 = y[n].fi;
    auto solve = [&](int x1, int x2, int y1, int y2)
    {
        int tx = x2 - x1 + 1, ty = y2 - y1 + 1;
        int res = tx * ty;
        if (res < n)
            return res + min(tx, ty);
        else
            return res;
    };
    int ans = INF;
    ff(i, 1, n)
    {
        int l = min1, r = max1;
        int u = min2, d = max2;
        if (x[1].se == i)
            l = x[min(n, 2LL)].fi;
        else if (x[n].se == i)
            r = x[max(1LL, n - 1)].fi;
        if (y[1].se == i)
            u = y[min(n, 2LL)].fi;
        else if (y[n].se == i)
            d = y[max(1LL, n - 1)].fi;

        ans = min(ans, solve(l, r, u, d));
    }
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    //
    cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}