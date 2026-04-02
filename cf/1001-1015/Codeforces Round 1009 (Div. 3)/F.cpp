#include <bits/stdc++.h>
// Finish Time: 2025/3/12 15:00:44 AC
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
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
int x11, y11, x22, y22;
int ans = 0;
bool ok = true;
void ss(int num)
{
    int len1 = y11 - x11, len2 = y22 - x22;
    if (y11 == x11 || y22 == x22)
    {
        ok = false;
        return;
    }
    if (x11 % num == num / 2)
    {
        x11 += num / 2;
        ans += len2 / (num / 2);
        len1 -= num / 2;
    }
    if (y11 == x11 || y22 == x22)
    {
        ok = false;
        return;
    }
    if ((y11 - x11) % num == num / 2)
    {
        y11 -= num / 2;
        ans += len2 / (num / 2);
        len1 -= num / 2;
    }
    if (y11 == x11 || y22 == x22)
    {
        ok = false;
        return;
    }
    if (x22 % num == num / 2)
    {
        x22 += num / 2;
        ans += len1 / (num / 2);
        len2 -= num / 2;
    }
    if (y11 == x11 || y22 == x22)
    {
        ok = false;
        return;
    }
    if ((y22 - x22) % num == num / 2)
    {
        y22 -= num / 2;
        ans += len1 / (num / 2);
        len2 -= num / 2;
    }
    if (y11 == x11 || y22 == x22)
    {
        ok = false;
        return;
    }
}
void solve()
{
    ans = 0;
    ok = true;
    cin >> x11 >> y11 >> x22 >> y22;
    for (int i = 1; ok; i++)
        ss(1LL << i);
    cout << ans << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}