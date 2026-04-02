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
const int N = 1e6 + 6;
void Murasame()
{
    int n, a, b;
    cin >> n >> a >> b;
    auto f = [&](int y) -> int
    {
        int x = n - y;
        return x * (a + b * y) + y;
    };
    auto g = [&](int x) -> int
    {
        if (x <= 0)
            return 0;
        if (x >= n)
            return n;
        return x;
    };
    if (b == 0)
    {
        cout << max(a * n, n) << endl;
    }
    else
    {
        double t = double(b * n - a + 1) / (2.0 * b);
        int x = floor(t);
        int y = x + 1;
        x = g(x);
        y = g(y);
        cout << max(f(x), f(y)) << endl;
    }
    // cout << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    // cin >> _T;
    while (_T--)
    {
        Murasame();
    }
    return 0;
}