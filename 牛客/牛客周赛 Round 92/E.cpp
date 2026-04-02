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
void Murasame()
{
    int n;
    cin >> n;
    vi a(n + 1);
    int sum = 0;
    ff(i, 1, n) cin >> a[i], sum += a[i];
    vi g[sum + 1];
    int now = 2;
    ff(i, 2, n)
    {
        int need = a[i];
        int bg = now - a[i - 1];
        ff(j, bg, bg + a[i - 1] - 1)
        {
            while (g[j].size() < 2 && need)
            {
                g[j].pb(now);
                now++;
                need--;
            }
        }
    }
    cout << 1 << endl;
    ff(i, 1, now - 1)
    {
        int cnt = 0;
        // cout << i << ": ";
        for (auto x : g[i])
            cout << x << " ", cnt++;
        while (cnt < 2)
            cout << -1 << " ", cnt++;
        cout << endl;
    }
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