#include <bits/stdc++.h>
// Finish Time: 2025/3/20 15:31:59 AC
#define i64 long long
#define u64 unsigned long long
#define i128 __int128
#define d64 long double
#define ff(x, y, z) for (int(x) = (y); (x) < (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) > (z); --(x))
#define pb push_back
#define eb emplace_back
#define vc vector
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
int h[1000005];
int cnt[1000005] ;
void solve()
{
    int n, x, c = 0;
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        cin >> h[i];
        for (int j = 1; j * j <= h[i]; j++)
        {
            if (h[i] % j == 0)
            {
                cnt[j]++;
                if (j * j != h[i])
                    cnt[h[i] / j]++;
            }
        }
    }
    for (int i = 1000000; i >= 1; i--)
    {
        if (cnt[i] >= 3)
        {
            x = i;
            break;
        }
    }
    sort(h + 1, h + n + 1);
    for (int i = 1; i <= n; i++)
    {
        if (h[i] % x == 0)
        {
            cout << h[i] << " ";
            c++;
        }
        if (c == 3)
            return;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    // cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}