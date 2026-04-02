#define _CRT_SECURE_NO_WARNINGS 1
#include <bits/stdc++.h>
#define int long long
#define ld long double
#define ull unsigned long long
#define lowbit(x) (x & -x)
#define pb push_back
#define pii pair<int, int>
#define mp make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define endl '\n'
using namespace std;
int a[200005];
void solve()
{
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        cin >> a[i];
        if (i > 1)
        {
            int temp = min(a[i - 1], a[i]);
            a[i - 1] -= temp;
            a[i] -= temp;
        }
    }
    for (int i = 1; i < n; i++)
    {
        if (a[i] > a[i + 1])
        {
            cout << "No" << endl;
            return;
        }
    }
    cout << "Yes" << endl;
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