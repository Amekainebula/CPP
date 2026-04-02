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
void Murasame()
{
    int n, m, x, y;
    cin >> n >> m >> x >> y;
    vc<pii> v = {{x, m}, {n - x + 1, m}, {n, y}, {n, m - y + 1}};
    int ans = n + m;
    for (auto [a, b] : v)
    {
        int temp = 0;
        while (a > 1)
        {
            temp++;
            a = (a + 1) / 2;
        }
        while (b > 1)
        {
            temp++;
            b = (b + 1) / 2;
        }
        ans = min(ans, temp);
    }
    cout << ans+1 << endl;
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