#include <bits/stdc++.h>
#define int long long
#define ull unsigned long long
#define i128 __int128
#define ld long double
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
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
void solve()
{
    int n;
    cin >> n;
    string s;
    cin >> s;
    int ans = 0;
    stack<char> q;
    for (int i = 0; i < n; i++)
    {
        if (q.empty() || s[i] != q.top()) 
        {
            q.push(s[i]);
        }
        else 
        {
            q.pop();
        }
    }
    cout << q.size() / 2 << endl;
}
// if (n % 2 == 0)
// {

// else
// {
//     int ans1 = 0, ans2 = 0;
//     n--;
//     for (int i = 0; i < n / 2; i++)
//     {
//         // cout << i << " " << n - i - 1 << endl;
//         if (s[i] != s[n - i - 1])
//             ans1++;
//     }
//     n++;
//     for (int i = 1; i <= n / 2; i++)
//     {
//         // cout << i << " " << n - i << endl;
//         if (s[i] != s[n - i])
//             ans2++;
//     }
//     ans = min(ans1, ans2);
//     cout << ans << endl;
// }
// cout << ans << endl;

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int _T = 1;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}